"""
Identificação de função de transferência a partir de um único Bode Plot
experimental, com busca automática de estrutura (Optuna + AICc) e ajuste
final via otimização não-linear (scipy.optimize.least_squares).

Combina:
  - a busca automática de estrutura (Optuna + AICc)
  - a parametrização física em frequência natural (wn, rad/s) e
    amortecimento (zeta), na mesma escala dos dados medidos no Bode

Estratégia de x0 (chute inicial):
  Como o Optuna testa uma estrutura diferente a cada trial, não dá pra
  fixar chutes manuais (como se faz olhando um único Bode). Em vez
  disso, os chutes de wn para cada polo/zero são espalhados
  logaritmicamente ao longo da faixa de frequência dos próprios dados
  (omega.min() a omega.max()).

Autor original: Edélio Gabriel Magalhães de Jesus
Refatorado para estrutura orientada a objetos com auxílio de IA.

Uso:

python optuna_tf_identifier.py --bode-file "./bode_files_PAPU/Id_1_Vel_NC_kp_586_Tn_15__1.csv" --system-part "Open-Loop" --n-trials 200 --output-dir "./tfs_json_PAPU" --plot-dir "./bodes_adjust_results"
"""

import argparse
import io
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import control as ct
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import scipy.optimize as opt

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


# =============================================================================
# Estrutura do modelo (quantos polos/zeros de cada tipo) — decidida pelo Optuna
# =============================================================================

@dataclass
class StructureConfig:
    """Estrutura de um modelo de função de transferência: quantos polos/zeros
    reais e complexos, e presença de integrador/diferenciador na origem.

    Ao contrário do ModelConfig de tf_identifier.py, aqui não há x0/bounds
    fixos por polo/zero -- eles são recalculados a cada avaliação a partir
    da faixa de frequência dos dados (ver TransferFunctionModel._build_x0).
    """

    n_zeros_reais: int = 0
    n_pares_zeros_complexos: int = 0
    n_polos_reais: int = 0
    n_pares_polos_complexos: int = 0
    tem_zero_origem: bool = False
    tem_polo_origem: bool = True

    @property
    def grau_num(self) -> int:
        return self.n_zeros_reais + 2 * self.n_pares_zeros_complexos + (1 if self.tem_zero_origem else 0)

    @property
    def grau_den(self) -> int:
        return self.n_polos_reais + 2 * self.n_pares_polos_complexos + (1 if self.tem_polo_origem else 0)

    @property
    def n_params(self) -> int:
        """Número de parâmetros contínuos do modelo (inclui o ganho K)."""
        return (
            1
            + self.n_zeros_reais
            + 2 * self.n_pares_zeros_complexos
            + self.n_polos_reais
            + 2 * self.n_pares_polos_complexos
        )

    def is_valida(self) -> bool:
        """Uma TF própria precisa de grau(num) <= grau(den) e ao menos um polo."""
        return self.grau_num <= self.grau_den and self.grau_den > 0

    @classmethod
    def from_optuna_params(cls, params: dict) -> "StructureConfig":
        return cls(
            n_zeros_reais=params["n_zeros_reais"],
            n_pares_zeros_complexos=params["n_zeros_complexos"],
            n_polos_reais=params["n_polos_reais"],
            n_pares_polos_complexos=params["n_polos_complexos"],
            tem_zero_origem=params["tem_zero_origem"],
            tem_polo_origem=params["tem_polo_origem"],
        )


@dataclass
class SearchBounds:
    """Bounds de frequência natural (rad/s) e amortecimento (zeta) usados
    durante a busca de estrutura. wn_min/wn_max são recalculados a partir
    dos dados (uma margem de décadas acima/abaixo da faixa medida)."""

    wn_min: float
    wn_max: float
    zeta_min: float = 0.01
    zeta_max_zero: float = 1.0
    zeta_max_polo: float = 2.0

    @classmethod
    def from_omega(cls, omega: np.ndarray, margem_decadas: float = 0.3) -> "SearchBounds":
        return cls(
            wn_min=omega.min() / (10 ** margem_decadas),
            wn_max=omega.max() * (10 ** margem_decadas),
        )


# =============================================================================
# Modelo de função de transferência: monta resíduos, ajusta, converte formas
# =============================================================================

class TransferFunctionModel:
    """
    Representa um modelo paramétrico de função de transferência (polos e
    zeros reais/complexos, na forma frequência-natural/amortecimento) para
    uma dada StructureConfig, e a lógica para ajustá-lo a dados de resposta
    em frequência via scipy.optimize.least_squares.

    Não conhece arquivos, plots, Optuna ou exportação -- só a estrutura
    matemática de uma estrutura fixa de polos/zeros.
    """

    def __init__(self, structure: StructureConfig, bounds: SearchBounds):
        self.structure = structure
        self.bounds = bounds
        self.result: Optional[opt.OptimizeResult] = None

        # Preenchidos após fit()
        self.K_bode: Optional[float] = None
        self.wn_z_real_opt: np.ndarray = np.array([])
        self.params_z_comp_opt: np.ndarray = np.array([])
        self.wn_p_real_opt: np.ndarray = np.array([])
        self.params_p_comp_opt: np.ndarray = np.array([])

    # ---------- API pública ----------

    def fit(self, omega: np.ndarray, frdata: np.ndarray, **least_squares_kwargs) -> opt.OptimizeResult:
        """Ajusta o modelo aos dados (omega em rad/s, frdata = resposta complexa)."""
        x0 = self._build_x0(omega)
        bounds = self._build_bounds()

        kwargs = dict(x_scale="jac", max_nfev=3000)
        kwargs.update(least_squares_kwargs)

        self.result = opt.least_squares(
            lambda params: self._residuals(params, omega, frdata),
            x0, bounds=bounds, **kwargs,
        )
        self._unpack_result(self.result.x)
        return self.result

    def aicc(self, n_amostras: int) -> float:
        """Critério de informação de Akaike corrigido, calculado a partir do
        último resultado de fit(). Usado pelo StructureSearcher para comparar
        estruturas de complexidade diferente."""
        self._check_fitted()
        k_params = self.structure.n_params

        if n_amostras - k_params - 1 <= 0:
            return np.inf

        custo_rss = max(np.sum(self.result.fun ** 2), 1e-12)
        aic = n_amostras * np.log(custo_rss / n_amostras) + 2 * k_params
        return aic + (2 * k_params * (k_params + 1)) / (n_amostras - k_params - 1)

    def to_tf(self) -> ct.TransferFunction:
        """Retorna a função de transferência ajustada na forma polinomial (Num/Den)."""
        self._check_fitted()
        cfg = self.structure

        num_poly = np.array([self.K_bode])
        for wz in self.wn_z_real_opt:
            num_poly = np.convolve(num_poly, [1.0 / wz, 1.0])
        for i in range(cfg.n_pares_zeros_complexos):
            wn_z = self.params_z_comp_opt[2 * i]
            zz = self.params_z_comp_opt[2 * i + 1]
            num_poly = np.convolve(num_poly, [1.0 / wn_z**2, 2 * zz / wn_z, 1.0])
        if cfg.tem_zero_origem:
            num_poly = np.convolve(num_poly, [1.0, 0.0])

        den_poly = np.array([1.0])
        for wp in self.wn_p_real_opt:
            den_poly = np.convolve(den_poly, [1.0 / wp, 1.0])
        for i in range(cfg.n_pares_polos_complexos):
            wn_p = self.params_p_comp_opt[2 * i]
            zp = self.params_p_comp_opt[2 * i + 1]
            den_poly = np.convolve(den_poly, [1.0 / wn_p**2, 2 * zp / wn_p, 1.0])
        if cfg.tem_polo_origem:
            den_poly = np.convolve(den_poly, [1.0, 0.0])

        return ct.tf(num_poly, den_poly)

    def to_zpk(self) -> ct.TransferFunction:
        """Retorna a função de transferência ajustada na forma ZPK (Zero-Pole-Gain)."""
        self._check_fitted()
        cfg = self.structure

        zeros_zpk: List[complex] = list(-self.wn_z_real_opt)
        for i in range(cfg.n_pares_zeros_complexos):
            wn_z = self.params_z_comp_opt[2 * i]
            zz = self.params_z_comp_opt[2 * i + 1]
            zeros_zpk.extend(np.roots([1.0, 2 * zz * wn_z, wn_z**2]))

        polos_zpk: List[complex] = list(-self.wn_p_real_opt)
        for i in range(cfg.n_pares_polos_complexos):
            wn_p = self.params_p_comp_opt[2 * i]
            zp = self.params_p_comp_opt[2 * i + 1]
            polos_zpk.extend(np.roots([1.0, 2 * zp * wn_p, wn_p**2]))

        if cfg.tem_zero_origem:
            zeros_zpk.append(0.0)
        if cfg.tem_polo_origem:
            polos_zpk.append(0.0)

        fator_num = (
            np.prod(1.0 / self.wn_z_real_opt) if cfg.n_zeros_reais > 0 else 1.0
        ) * (
            np.prod(1.0 / self.params_z_comp_opt[0::2] ** 2)
            if cfg.n_pares_zeros_complexos > 0 else 1.0
        )
        fator_den = (
            np.prod(1.0 / self.wn_p_real_opt) if cfg.n_polos_reais > 0 else 1.0
        ) * (
            np.prod(1.0 / self.params_p_comp_opt[0::2] ** 2)
            if cfg.n_pares_polos_complexos > 0 else 1.0
        )
        K_zpk = self.K_bode * (fator_num / fator_den)

        return ct.zpk(zeros_zpk, polos_zpk, K_zpk)

    def summary(self) -> str:
        """Texto legível com a estrutura, os parâmetros otimizados e os
        diagnósticos da otimização final."""
        self._check_fitted()
        cfg = self.structure
        lines = [
            "=== ESTRUTURA ===",
            f"n_zeros_reais={cfg.n_zeros_reais}, n_zeros_complexos={cfg.n_pares_zeros_complexos}, "
            f"n_polos_reais={cfg.n_polos_reais}, n_polos_complexos={cfg.n_pares_polos_complexos}, "
            f"tem_zero_origem={cfg.tem_zero_origem}, tem_polo_origem={cfg.tem_polo_origem}",
            "",
            "=== RESULTADO DA OTIMIZAÇÃO ===",
            f"Status: {self.result.status} | {self.result.message}",
            f"Nº avaliações: {self.result.nfev}",
            f"Custo final: {self.result.cost:.6e}",
            "",
            "=== PARÂMETROS OTIMIZADOS (forma wn/zeta) ===",
            f"K (Ganho Bode) = {self.K_bode:.4f}",
        ]
        for i, wz in enumerate(self.wn_z_real_opt, 1):
            lines.append(f"Zero Real {i}: wn_z = {wz:.2f} rad/s | z = {-wz:.2f} rad/s")
        for i in range(cfg.n_pares_zeros_complexos):
            wn_z = self.params_z_comp_opt[2 * i]
            zz = self.params_z_comp_opt[2 * i + 1]
            lines.append(
                f"Par de Zeros Complexos {i+1}: wn = {wn_z:.2f} rad/s "
                f"({wn_z/(2*np.pi):.2f} Hz) | zeta = {zz:.4f}"
            )
        for i, wp in enumerate(self.wn_p_real_opt, 1):
            lines.append(f"Polo Real {i}: wn_p = {wp:.2f} rad/s | p = {-wp:.2f} rad/s")
        for i in range(cfg.n_pares_polos_complexos):
            wn_p = self.params_p_comp_opt[2 * i]
            zp = self.params_p_comp_opt[2 * i + 1]
            lines.append(
                f"Par de Polos Complexos {i+1}: wn = {wn_p:.2f} rad/s "
                f"({wn_p/(2*np.pi):.2f} Hz) | zeta = {zp:.4f}"
            )
        return "\n".join(lines)

    def rms_error(self) -> float:
        """Erro RMS do ajuste final, calculado a partir do result.cost do
        último fit(). result.fun concatena [erro_real, erro_imag], por isso
        o número de amostras é len(result.fun) // 2."""
        self._check_fitted()
        n_amostras = len(self.result.fun) // 2
        return float(np.sqrt(2 * self.result.cost / n_amostras))

    def to_json_dict(self) -> dict:
        """Coeficientes Num/Den da TF ajustada + estrutura, prontos para
        exportação em JSON."""
        sys_tf = self.to_tf()
        cfg = self.structure
        return {
            "num": [float(c) for c in sys_tf.num[0][0]],
            "den": [float(c) for c in sys_tf.den[0][0]],
            "dt": sys_tf.dt if sys_tf.dt is not None else 0,
            "rms_error": self.rms_error(),
            "estrutura": {
                "n_zeros_reais": cfg.n_zeros_reais,
                "n_zeros_complexos": cfg.n_pares_zeros_complexos,
                "n_polos_reais": cfg.n_polos_reais,
                "n_polos_complexos": cfg.n_pares_polos_complexos,
                "tem_zero_origem": cfg.tem_zero_origem,
                "tem_polo_origem": cfg.tem_polo_origem,
            },
        }

    # ---------- Métodos internos ----------

    def _check_fitted(self) -> None:
        if self.result is None:
            raise RuntimeError("O modelo ainda não foi ajustado. Chame fit() primeiro.")

    def _residuals(self, params: np.ndarray, omega: np.ndarray, frdata: np.ndarray) -> np.ndarray:
        cfg = self.structure
        K = params[0]
        idx = 1

        wn_z_real = params[idx: idx + cfg.n_zeros_reais]
        idx += cfg.n_zeros_reais

        params_z_comp = params[idx: idx + 2 * cfg.n_pares_zeros_complexos]
        idx += 2 * cfg.n_pares_zeros_complexos

        wn_p_real = params[idx: idx + cfg.n_polos_reais]
        idx += cfg.n_polos_reais

        params_p_comp = params[idx: idx + 2 * cfg.n_pares_polos_complexos]

        s = 1j * omega

        num_val = (s.copy() if cfg.tem_zero_origem else np.ones_like(s, dtype=complex)) * K
        for wz in wn_z_real:
            num_val *= (s / wz + 1.0)
        for i in range(cfg.n_pares_zeros_complexos):
            wn_z = params_z_comp[2 * i]
            zeta_z = params_z_comp[2 * i + 1]
            num_val *= (s**2 / wn_z**2 + 2 * zeta_z * s / wn_z + 1.0)

        den_val = s.copy() if cfg.tem_polo_origem else np.ones_like(s, dtype=complex)
        for wp in wn_p_real:
            den_val *= (s / wp + 1.0)
        for i in range(cfg.n_pares_polos_complexos):
            wn_p = params_p_comp[2 * i]
            zeta_p = params_p_comp[2 * i + 1]
            den_val *= (s**2 / wn_p**2 + 2 * zeta_p * s / wn_p + 1.0)

        H_est = num_val / den_val
        err = H_est - frdata
        return np.concatenate([err.real, err.imag])

    def _build_x0(self, omega: np.ndarray) -> List[float]:
        """Espalha os chutes de wn logaritmicamente entre omega.min() e
        omega.max(), na ordem: zeros reais, zeros complexos, polos reais,
        polos complexos. Evita que todos os parâmetros comecem no mesmo
        ponto (o que gerava mínimos locais diferentes a cada estrutura)."""
        cfg = self.structure
        n_total_wn = (
            cfg.n_zeros_reais + cfg.n_pares_zeros_complexos
            + cfg.n_polos_reais + cfg.n_pares_polos_complexos
        )

        if n_total_wn > 0:
            wn_guesses = np.logspace(np.log10(omega.min()), np.log10(omega.max()), n_total_wn)
        else:
            wn_guesses = np.array([])

        guess_iter = iter(wn_guesses)
        x0 = [1.0]

        for _ in range(cfg.n_zeros_reais):
            x0 += [next(guess_iter)]
        for _ in range(cfg.n_pares_zeros_complexos):
            x0 += [next(guess_iter), 0.5]
        for _ in range(cfg.n_polos_reais):
            x0 += [next(guess_iter)]
        for _ in range(cfg.n_pares_polos_complexos):
            x0 += [next(guess_iter), 0.5]

        return x0

    def _build_bounds(self) -> Tuple[List[float], List[float]]:
        cfg = self.structure
        b = self.bounds
        lower = [-np.inf]
        upper = [np.inf]

        for _ in range(cfg.n_zeros_reais):
            lower.append(b.wn_min)
            upper.append(b.wn_max)
        for _ in range(cfg.n_pares_zeros_complexos):
            lower.extend([b.wn_min, b.zeta_min])
            upper.extend([b.wn_max, b.zeta_max_zero])

        for _ in range(cfg.n_polos_reais):
            lower.append(b.wn_min)
            upper.append(b.wn_max)
        for _ in range(cfg.n_pares_polos_complexos):
            lower.extend([b.wn_min, b.zeta_min])
            upper.extend([b.wn_max, b.zeta_max_polo])

        return lower, upper

    def _unpack_result(self, x: np.ndarray) -> None:
        cfg = self.structure
        self.K_bode = x[0]
        idx = 1

        self.wn_z_real_opt = x[idx: idx + cfg.n_zeros_reais]
        idx += cfg.n_zeros_reais

        self.params_z_comp_opt = x[idx: idx + 2 * cfg.n_pares_zeros_complexos]
        idx += 2 * cfg.n_pares_zeros_complexos

        self.wn_p_real_opt = x[idx: idx + cfg.n_polos_reais]
        idx += cfg.n_polos_reais

        self.params_p_comp_opt = x[idx: idx + 2 * cfg.n_pares_polos_complexos]


# =============================================================================
# Busca de estrutura via Optuna: decide quantos polos/zeros usar, pelo AICc
# =============================================================================

class StructureSearcher:
    """
    Usa Optuna (TPESampler) para buscar, dentro de uma faixa de graus de
    polos/zeros reais e complexos, a estrutura de função de transferência
    que minimiza o AICc ao ser ajustada aos dados de resposta em frequência.

    Não conhece arquivos ou plots -- só recebe omega/frdata e devolve a
    melhor StructureConfig encontrada.
    """

    def __init__(
        self,
        omega: np.ndarray,
        frdata: np.ndarray,
        max_polos_reais: int = 2,
        max_zeros_reais: int = 2,
        max_polos_complexos: int = 2,
        max_zeros_complexos: int = 2,
        margem_decadas: float = 0.3,
        seed: int = 367,
    ):
        self.omega = omega
        self.frdata = frdata
        self.n_amostras = len(omega)

        self.max_polos_reais = max_polos_reais
        self.max_zeros_reais = max_zeros_reais
        self.max_polos_complexos = max_polos_complexos
        self.max_zeros_complexos = max_zeros_complexos

        self.bounds = SearchBounds.from_omega(omega, margem_decadas)
        self.seed = seed
        self.study: Optional[optuna.Study] = None

    # ---------- API pública ----------

    def search(self, n_trials: int = 500, n_jobs: int = -1, show_progress_bar: bool = True) -> StructureConfig:
        """Executa a busca de estrutura e retorna a melhor StructureConfig."""
        sampler = optuna.samplers.TPESampler(seed=self.seed)
        self.study = optuna.create_study(
            direction="minimize", sampler=sampler, study_name="polos_zeros_aicc_wn",
        )

        logger.info("Buscando a estrutura ideal de polos e zeros baseada no AICc...")
        self.study.optimize(
            self._objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=show_progress_bar,
        )

        logger.info("\nMelhores hiperparâmetros (estrutura):")
        for chave, valor in self.study.best_params.items():
            logger.info("  %s: %s", chave, valor)
        logger.info("\nMelhor score (AICc): %.6f", self.study.best_value)

        return StructureConfig.from_optuna_params(self.study.best_params)

    # ---------- Métodos internos ----------

    def _objective(self, trial: optuna.Trial) -> float:
        structure = StructureConfig(
            n_polos_reais=trial.suggest_int("n_polos_reais", 0, self.max_polos_reais),
            n_zeros_reais=trial.suggest_int("n_zeros_reais", 0, self.max_zeros_reais),
            n_pares_polos_complexos=trial.suggest_int("n_polos_complexos", 0, self.max_polos_complexos),
            n_pares_zeros_complexos=trial.suggest_int("n_zeros_complexos", 0, self.max_zeros_complexos),
            tem_polo_origem=trial.suggest_categorical("tem_polo_origem", [True, False]),
            tem_zero_origem=trial.suggest_categorical("tem_zero_origem", [True, False]),
        )

        if not structure.is_valida():
            raise optuna.exceptions.TrialPruned()

        if self.n_amostras - structure.n_params - 1 <= 0:
            raise optuna.exceptions.TrialPruned()

        model = TransferFunctionModel(structure, self.bounds)
        try:
            model.fit(self.omega, self.frdata, max_nfev=3000)
        except Exception:
            raise optuna.exceptions.TrialPruned()

        return model.aicc(self.n_amostras)


# =============================================================================
# Orquestração: carrega o Bode, busca a estrutura, ajusta, plota, exporta
# =============================================================================

class BodeStructureIdentifier:
    """
    Orquestra a identificação de função de transferência a partir de um
    único arquivo de Bode Plot, sem estrutura de modelo pré-definida: carrega
    os dados, busca a melhor estrutura de polos/zeros via StructureSearcher,
    refina o ajuste final com um TransferFunctionModel, gera o plot
    comparativo e exporta o resultado em JSON.

    Uso:
        identifier = BodeStructureIdentifier(
            bode_file="./bode_files_PAPU/Id_1_Pos_kp_10_Tn_0_20.csv",
            system_part="Process",
        )
        identifier.run(n_trials=500, output_dir="./tfs_json_PAPU", plot_dir="./bodes_pngs")
    """

    def __init__(
        self,
        bode_file: Union[str, Path],
        system_part: str = "Process",
        margem_decadas: float = 0.3,
        seed: int = 367,
    ):
        self.bode_file = Path(bode_file)
        self.system_part = system_part
        self.margem_decadas = margem_decadas
        self.seed = seed

        self.sys_frd: Optional[ct.FrequencyResponseData] = None
        self.omega: Optional[np.ndarray] = None
        self.frdata: Optional[np.ndarray] = None

        self.searcher: Optional[StructureSearcher] = None
        self.model: Optional[TransferFunctionModel] = None

    # ---------- API pública ----------

    def run(
        self,
        output_dir: Union[str, Path],
        plot_dir: Union[str, Path],
        n_trials: int = 500,
        n_jobs: int = -1,
    ) -> Path:
        """Executa o pipeline completo. Retorna o caminho do JSON exportado."""
        self._load_bode()

        self.searcher = StructureSearcher(
            self.omega, self.frdata, margem_decadas=self.margem_decadas, seed=self.seed,
        )
        best_structure = self.searcher.search(n_trials=n_trials, n_jobs=n_jobs)

        self.model = TransferFunctionModel(best_structure, self.searcher.bounds)
        self.model.fit(
            self.omega, self.frdata,
            max_nfev=20000, xtol=1e-12, ftol=1e-12,
        )
        logger.info("\n%s", self.model.summary())
        logger.info("Erro RMS: %.6e", self.model.rms_error())

        json_path = self._export_json(Path(output_dir))
        self._plot(Path(plot_dir))
        return json_path

    # ---------- Métodos internos ----------

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        with open(path, "r") as f:
            linhas = [linha.strip().strip('"') for linha in f]
        idx_header = next(i for i, l in enumerate(linhas) if l.startswith("Frequency"))
        texto_dados = "\n".join(linhas[idx_header:])
        return pd.read_csv(io.StringIO(texto_dados), sep="\t")

    def _load_bode(self) -> None:
        df = self._load_csv(self.bode_file)

        gain = df[f"{self.system_part}-Gain"].to_numpy()
        phase = df[f"{self.system_part}-Phase"].to_numpy()
        frequency = df["Frequency"].to_numpy()

        mag_linear = 10 ** (gain / 20)
        phase_rad = np.deg2rad(phase)
        omega = 2 * np.pi * frequency

        self.sys_frd = ct.frd(
            mag_linear * np.exp(1j * phase_rad), omega,
            name=f"{self.system_part} Bode Data",
        )
        self.omega = self.sys_frd.omega
        self.frdata = np.squeeze(self.sys_frd.frdata)

    def _base_output_name(self) -> str:
        return f"OPTUNA_{self.system_part}_TF_{self.bode_file.stem}"

    def _export_json(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{self._base_output_name()}.json"
        with open(json_path, "w") as f:
            json.dump(self.model.to_json_dict(), f, indent=4)
        logger.info("Função de transferência exportada para %s", json_path)
        return json_path

    def _plot(self, plot_dir: Path) -> None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        out = plot_dir / f"{self._base_output_name()}.png"

        sys_tf = self.model.to_tf()
        sys_zpk = self.model.to_zpk()

        ct.bode_plot(
            [self.sys_frd, sys_tf, sys_zpk], omega=self.omega, dB=True, Hz=True,
            label=["sys_frd", "sys_tf", "sys_zpk"], legend_loc="lower left",
        )
        plt.suptitle("Comparative Bode Plot")
        plt.savefig(out)
        logger.info("Saved %s", out)
        plt.close()


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> "argparse.Namespace":
    parser = argparse.ArgumentParser(
        description="Busca automaticamente (via Optuna + AICc) a estrutura de "
        "polos/zeros de uma função de transferência a partir de um único "
        "arquivo de Bode Plot experimental."
    )
    parser.add_argument("--bode-file", type=str, required=True, help="Arquivo .csv do Bode a analisar.")
    parser.add_argument(
        "--system-part", type=str, default="Process",
        choices=["Process", "Open-Loop", "Close-Loop"],
        help="Parte do sistema a identificar.",
    )
    parser.add_argument("--n-trials", type=int, default=500, help="Nº de trials do Optuna na busca de estrutura.")
    parser.add_argument("--output-dir", type=str, default="./tfs_json_PAPU", help="Pasta de saída do JSON.")
    parser.add_argument("--plot-dir", type=str, default="./bodes_pngs", help="Pasta de saída dos gráficos.")
    parser.add_argument("--show", action="store_true", help="Exibe os gráficos na tela além de salvar.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    identifier = BodeStructureIdentifier(
        bode_file=args.bode_file, system_part=args.system_part,
    )
    identifier.run(output_dir=args.output_dir, plot_dir=args.plot_dir, n_trials=args.n_trials)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()