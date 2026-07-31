"""
Identificação de função de transferência a partir de dados de Bode Plot
experimentais, via otimização não-linear (scipy.optimize.least_squares).

Autor original: Edélio Gabriel Magalhães de Jesus
Refatorado para estrutura orientada a objetos com auxílio de IA.

Uso:

 python tf_identifier.py --bode-file "./bode_files_PAPU/Id_1_Vel_NC_kp_586_Tn_15__1.csv" --model-config "./models_config_tfs_json/model_config_vel_nc_kp_586.json" --system-part "Open-Loop" --output-dir "./tfs_json_PAPU" --plot-dir "./bodes_pngs"
"""

import argparse
import io
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple, Union

import control as ct
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize as opt
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Estrutura do modelo (polos, zeros, x0, bounds) — carregada de um JSON
# =============================================================================

@dataclass
class BoundsConfig:
    wn_min: float = 1.0
    wn_max: float = 20e3
    wn_max_complexo: float = 2e3
    zeta_min: float = 0.1
    zeta_max: float = 1.0


@dataclass
class ModelConfig:
    """Estrutura de um modelo de função de transferência: quantos polos/zeros
    reais e complexos, presença de integrador/diferenciador na origem, e os
    palpites iniciais (frequências naturais, em rad/s) para cada um."""

    freq_natural_polos_comp: List[float] = field(default_factory=list)
    freq_natural_zeros_comp: List[float] = field(default_factory=list)
    freq_natural_polos_reais: List[float] = field(default_factory=list)
    freq_natural_zeros_reais: List[float] = field(default_factory=list)

    tem_polo_origem: bool = True
    tem_zero_origem: bool = False

    n_polos_reais: int = 0
    n_zeros_reais: int = 0
    n_pares_polos_complexos: int = 0
    n_pares_zeros_complexos: int = 0

    bounds: BoundsConfig = field(default_factory=BoundsConfig)

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "ModelConfig":
        with open(path, "r") as f:
            data = json.load(f)

        bounds_data = data.pop("bounds", {})
        return cls(**data, bounds=BoundsConfig(**bounds_data))

    def to_json(self, path: Union[str, Path]) -> None:
        data = {
            "freq_natural_polos_comp": self.freq_natural_polos_comp,
            "freq_natural_zeros_comp": self.freq_natural_zeros_comp,
            "freq_natural_polos_reais": self.freq_natural_polos_reais,
            "freq_natural_zeros_reais": self.freq_natural_zeros_reais,
            "tem_polo_origem": self.tem_polo_origem,
            "tem_zero_origem": self.tem_zero_origem,
            "n_polos_reais": self.n_polos_reais,
            "n_zeros_reais": self.n_zeros_reais,
            "n_pares_polos_complexos": self.n_pares_polos_complexos,
            "n_pares_zeros_complexos": self.n_pares_zeros_complexos,
            "bounds": vars(self.bounds),
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=4)


# =============================================================================
# Modelo de função de transferência: monta resíduos, otimiza, converte formas
# =============================================================================

class TransferFunctionModel:
    """
    Representa um modelo paramétrico de função de transferência (polos e
    zeros reais/complexos, na forma frequência-natural/amortecimento) e a
    lógica para ajustá-lo a dados de resposta em frequência via
    scipy.optimize.least_squares.

    Não conhece arquivos, plots ou exportação — só a estrutura matemática.
    """

    def __init__(self, config: ModelConfig):
        self.config = config
        self.result: Optional[opt.OptimizeResult] = None

        # Preenchidos após fit()
        self.K_bode: Optional[float] = None
        self.wn_z_real_opt: np.ndarray = np.array([])
        self.params_z_comp_opt: np.ndarray = np.array([])
        self.wn_p_real_opt: np.ndarray = np.array([])
        self.params_p_comp_opt: np.ndarray = np.array([])

    # ---------- API pública ----------

    def fit(self, omega: np.ndarray, frdata: np.ndarray) -> opt.OptimizeResult:
        """Ajusta o modelo aos dados (omega em rad/s, frdata = resposta complexa)."""
        x0 = self._build_x0()
        bounds = self._build_bounds()

        self.result = opt.least_squares(
            lambda params: self._residuals(params, omega, frdata),
            x0, bounds=bounds, x_scale="jac",
        )
        self._unpack_result(self.result.x)
        return self.result

    def to_tf(self) -> ct.TransferFunction:
        """Retorna a função de transferência ajustada na forma polinomial (Num/Den)."""
        self._check_fitted()
        cfg = self.config

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
        cfg = self.config

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
        """Texto legível com os parâmetros otimizados e diagnósticos da otimização."""
        self._check_fitted()
        lines = [
            "=== RESULTADO DA OTIMIZAÇÃO ===",
            f"Status: {self.result.status} | {self.result.message}",
            f"Nº avaliações: {self.result.nfev}",
            f"Custo final: {self.result.cost:.6e}",
            "",
            "=== PARÂMETROS OTIMIZADOS ===",
            f"K (Ganho Bode) = {self.K_bode:.4f}",
            f"Zero na origem: {self.config.tem_zero_origem} | "
            f"Polo na origem: {self.config.tem_polo_origem}",
        ]
        for i, wz in enumerate(self.wn_z_real_opt, 1):
            lines.append(f"Zero Real {i}: wn_z = {wz:.2f} rad/s")
        for i in range(self.config.n_pares_zeros_complexos):
            wn_z = self.params_z_comp_opt[2 * i]
            zz = self.params_z_comp_opt[2 * i + 1]
            lines.append(f"Zero Complexo {i+1}: wn = {wn_z:.2f} rad/s | zeta = {zz:.4f}")
        for i, wp in enumerate(self.wn_p_real_opt, 1):
            lines.append(f"Polo Real {i}: wn_p = {wp:.2f} rad/s")
        for i in range(self.config.n_pares_polos_complexos):
            wn_p = self.params_p_comp_opt[2 * i]
            zp = self.params_p_comp_opt[2 * i + 1]
            lines.append(f"Polo Complexo {i+1}: wn = {wn_p:.2f} rad/s | zeta = {zp:.4f}")
        return "\n".join(lines)

    def to_json_dict(self) -> dict:
        """Coeficientes Num/Den da TF ajustada, prontos para exportação em JSON."""
        sys_tf = self.to_tf()
        return {
            "num": [float(c) for c in sys_tf.num[0][0]],
            "den": [float(c) for c in sys_tf.den[0][0]],
            "dt": sys_tf.dt if sys_tf.dt is not None else 0,
        }

    # ---------- Métodos internos ----------

    def _check_fitted(self) -> None:
        if self.result is None:
            raise RuntimeError("O modelo ainda não foi ajustado. Chame fit() primeiro.")

    def _residuals(self, params: np.ndarray, omega: np.ndarray, frdata: np.ndarray) -> np.ndarray:
        cfg = self.config
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

    def _build_x0(self) -> List[float]:
        cfg = self.config
        x0 = [1.0]

        for _, f in zip(range(cfg.n_zeros_reais), cfg.freq_natural_zeros_reais):
            x0 += [f]
        for _, f in zip(range(cfg.n_pares_zeros_complexos), cfg.freq_natural_zeros_comp):
            x0 += [f, 0.5]
        for _, f in zip(range(cfg.n_polos_reais), cfg.freq_natural_polos_reais):
            x0 += [f]
        for _, f in zip(range(cfg.n_pares_polos_complexos), cfg.freq_natural_polos_comp):
            x0 += [f, 0.5]

        return x0

    def _build_bounds(self) -> Tuple[List[float], List[float]]:
        cfg = self.config
        b = cfg.bounds
        lower = [-np.inf]
        upper = [np.inf]

        for _ in range(cfg.n_zeros_reais):
            lower.append(b.wn_min)
            upper.append(b.wn_max)
        for _ in range(cfg.n_pares_zeros_complexos):
            lower.extend([b.wn_min, b.zeta_min])
            upper.extend([b.wn_max_complexo, b.zeta_max])

        for _ in range(cfg.n_polos_reais):
            lower.append(b.wn_min)
            upper.append(b.wn_max)
        for _ in range(cfg.n_pares_polos_complexos):
            lower.extend([b.wn_min, b.zeta_min])
            upper.extend([b.wn_max_complexo, b.zeta_max])

        return lower, upper

    def _unpack_result(self, x: np.ndarray) -> None:
        cfg = self.config
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
# Orquestração: carrega o Bode, ajusta o modelo, plota, exporta
# =============================================================================

class FRFIdentifier:
    """
    Orquestra a identificação de função de transferência a partir de um único
    arquivo de Bode Plot: carrega os dados, ajusta um TransferFunctionModel,
    gera os plots de diagnóstico e exporta o resultado em JSON.

    Uso:
        identifier = FRFIdentifier(
            bode_file="./bode_files_PAPU/Id_1_kp_10_Tn_0_1.csv",
            model_config=ModelConfig.from_json("./model_2pol_1zero.json"),
            system_part="Open-Loop",
        )
        identifier.run(output_dir="./tfs_json_PAPU", plot_dir="./bodes_pngs")

        
    """

    def __init__(
        self,
        bode_file: Union[str, Path],
        model_config: ModelConfig,
        system_part: str = "Open-Loop",
    ):
        self.bode_file = Path(bode_file)
        self.system_part = system_part
        self.model = TransferFunctionModel(model_config)

        self.sys_frd: Optional[ct.FrequencyResponseData] = None
        self.omega: Optional[np.ndarray] = None
        self.frdata: Optional[np.ndarray] = None

    # ---------- API pública ----------

    def run(self, output_dir: Union[str, Path], plot_dir: Union[str, Path]) -> Path:
        """Executa o pipeline completo. Retorna o caminho do JSON exportado."""
        self._load_bode()
        self.model.fit(self.omega, self.frdata)
        logger.info("\n%s", self.model.summary())

        json_path = self._export_json(Path(output_dir))
        self._plot_all(Path(plot_dir))
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
        return f"LS_{self.system_part.lower()}_TF_{self.bode_file.stem}"

    def _export_json(self, output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = output_dir / f"{self._base_output_name()}.json"
        with open(json_path, "w") as f:
            json.dump(self.model.to_json_dict(), f, indent=4)
        logger.info("Função de transferência exportada para %s", json_path)
        return json_path

    def _plot_all(self, plot_dir: Path) -> None:
        plot_dir.mkdir(parents=True, exist_ok=True)
        base_path = plot_dir / self._base_output_name()

        sys_tf = self.model.to_tf()
        sys_zpk = self.model.to_zpk()

        self._plot_dominio_dados(base_path, sys_tf, sys_zpk)
        omega_ext = self._plot_dominio_expandido(base_path, sys_tf, sys_zpk)
        self._plot_com_margens(base_path, sys_tf, omega_ext)

    def _plot_dominio_dados(self, base_path: Path, sys_tf, sys_zpk) -> None:
        ct.bode_plot(
            [self.sys_frd, sys_tf, sys_zpk], omega=self.omega, dB=True, Hz=True,
            label=["sys_frd", "sys_tf", "sys_zpk"], legend_loc="lower left",
        )
        plt.suptitle("Comparative Bode Plot")
        out = f"{base_path}_DOMINIO_DADOS.png"
        plt.savefig(out)
        logger.info("Saved %s", out)
        plt.close()

    def _plot_dominio_expandido(self, base_path: Path, sys_tf, sys_zpk) -> np.ndarray:
        omega_dados = self.sys_frd.omega
        omega_min_ext = omega_dados.min() / 1000
        omega_max_ext = omega_dados.max() * 10
        omega_ext = np.logspace(np.log10(omega_min_ext), np.log10(omega_max_ext), 500)

        resp_tf = ct.frequency_response(sys_tf, omega_ext)
        resp_zpk = ct.frequency_response(sys_zpk, omega_ext)

        mag_tf_db = 20 * np.log10(np.abs(resp_tf.frdata[0, 0]))
        mag_zpk_db = 20 * np.log10(np.abs(resp_zpk.frdata[0, 0]))
        fase_tf_deg = np.rad2deg(np.unwrap(np.angle(resp_tf.frdata[0, 0])))
        fase_zpk_deg = np.rad2deg(np.unwrap(np.angle(resp_zpk.frdata[0, 0])))

        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        ax_mag.semilogx(omega_dados, 20 * np.log10(np.abs(self.frdata)), "o", label="Dados (FRD)", markersize=4)
        ax_mag.semilogx(omega_ext, mag_tf_db, label="Modelo TF")
        ax_mag.semilogx(omega_ext, mag_zpk_db, "--", label="Modelo ZPK")
        ax_mag.set_ylabel("Magnitude (dB)")
        ax_mag.legend()
        ax_mag.grid(True, which="both")

        ax_phase.semilogx(omega_dados, np.rad2deg(np.unwrap(np.angle(self.frdata))), "o", markersize=4)
        ax_phase.semilogx(omega_ext, fase_tf_deg)
        ax_phase.semilogx(omega_ext, fase_zpk_deg, "--")
        ax_phase.set_ylabel("Fase (graus)")
        ax_phase.set_xlabel("Frequência (rad/s)")
        ax_phase.grid(True, which="both")

        pad_mag, pad_fase = 5, 15
        ax_mag.set_ylim(
            min(mag_tf_db.min(), mag_zpk_db.min()) - pad_mag,
            max(mag_tf_db.max(), mag_zpk_db.max()) + pad_mag,
        )
        ax_phase.set_ylim(
            min(fase_tf_deg.min(), fase_zpk_deg.min()) - pad_fase,
            max(fase_tf_deg.max(), fase_zpk_deg.max()) + pad_fase,
        )

        plt.tight_layout()
        out = f"{base_path}_DOMINIO_EXPANDIDO.png"
        plt.savefig(out)
        logger.info("Saved %s", out)
        plt.close()

        return omega_ext

    def _plot_com_margens(self, base_path: Path, sys_tf, omega_ext: np.ndarray) -> None:
        if self.system_part != "Open-Loop":
            return

        gm, pm, wcg, wcp = ct.margin(self.sys_frd)

        if np.isnan(gm) or np.isnan(pm) or np.isinf(gm) or np.isinf(pm):
            logger.info("Margens não encontradas ou infinitas no FRD. Plotando a TF estimada...")
            ct.bode_plot(sys_tf, omega=omega_ext, dB=True, Hz=False, display_margins=True)
            plt.suptitle("Estimated Transfer Function Bode Plot")
            out = f"{base_path}_DOMINIO_EXPANDIDO_COM_MARGENS.png"
        else:
            logger.info("Margens válidas encontradas no FRD. Plotando os dados medidos...")
            ct.bode_plot(self.sys_frd, dB=True, Hz=True, display_margins=True)
            plt.suptitle("Transfer Function Bode Plot")
            out = f"{base_path}_DOMINIO_DADOS_COM_MARGENS.png"

        plt.savefig(out)
        logger.info("Saved %s", out)
        plt.close()


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> "argparse.Namespace":
    parser = argparse.ArgumentParser(
        description="Identifica a função de transferência de um sistema a "
        "partir de um único arquivo de Bode Plot experimental."
    )
    parser.add_argument("--bode-file", type=str, required=True, help="Arquivo .csv do Bode a analisar.")
    parser.add_argument(
        "--model-config", type=str, required=True,
        help="Arquivo JSON com a estrutura do modelo (polos, zeros, x0, bounds).",
    )
    parser.add_argument(
        "--system-part", type=str, default="Open-Loop",
        choices=["Process", "Open-Loop", "Close-Loop"],
        help="Parte do sistema a identificar.",
    )
    parser.add_argument("--output-dir", type=str, default="./tfs_json_PAPU", help="Pasta de saída do JSON.")
    parser.add_argument("--plot-dir", type=str, default="./bodes_pngs", help="Pasta de saída dos gráficos.")
    parser.add_argument("--show", action="store_true", help="Exibe os gráficos na tela além de salvar.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config = ModelConfig.from_json(args.model_config)
    identifier = FRFIdentifier(
        bode_file=args.bode_file, model_config=config, system_part=args.system_part,
    )
    identifier.run(output_dir=args.output_dir, plot_dir=args.plot_dir)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()