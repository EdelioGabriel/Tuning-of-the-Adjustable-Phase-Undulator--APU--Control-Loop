"""
Identificação de função de transferência a partir de dados de Bode Plot
experimentais, via Vector Fitting (scikit-rf).

Diferente da versão paramétrica (frf_identifier.py, que usa
scipy.optimize.least_squares com um modelo polo/zero "à mão"), esta versão
usa o algoritmo de Vector Fitting (Gustavsen & Semlyen) para estimar
automaticamente polos, resíduos e ganho a partir apenas da quantidade de
polos desejada — sem exigir um palpite inicial (wn, zeta) fisicamente
informado.

Referência: https://scikit-rf.readthedocs.io/en/latest/tutorials/VectorFitting.html

Autor original: Edélio Gabriel Magalhães de Jesus
Refatorado para estrutura orientada a objetos com auxílio de IA.

Uso:

python vector_fitting_identifier.py --bode-file "./bode_files_PAPU/Id_1_Vel_NC_kp_586_Tn_15__1.csv" --vf-config "./models_config_tfs_json/model_vf_config_vel_nc_kp_586.json" --system-part "Open-Loop" --output-dir "./tfs_json_PAPU" --plot-dir "./bodes_adjust_results"

"""

import argparse
import io
import itertools
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple, Union

import control as ct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.signal as sig
import skrf
from skrf.vectorFitting import VectorFitting

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Configuração da ordem do modelo de Vector Fitting
# =============================================================================

@dataclass
class VectorFittingConfig:
    """Configuração da ordem do modelo e dos parâmetros de convergência do
    Vector Fitting.

    Diferente do ModelConfig usado no ajuste paramétrico (frf_identifier.py),
    aqui não se define wn/zeta iniciais — só a quantidade de polos; o próprio
    algoritmo distribui e refina as posições.
    """

    n_polos_reais: int = 2
    n_pares_polos_complexos: int = 1
    tem_polo_origem: bool = True

    max_iteracoes: int = 100
    tolerancia: float = 1e-3
    forcar_passividade: bool = False

    @classmethod
    def from_json(cls, path: Union[str, Path]) -> "VectorFittingConfig":
        with open(path, "r") as f:
            data = json.load(f)
        return cls(**data)

    def to_json(self, path: Union[str, Path]) -> None:
        with open(path, "w") as f:
            json.dump(vars(self), f, indent=4)

    @property
    def n_polos_reais_total(self) -> int:
        """Polos reais livres + 1 polo próximo da origem, se configurado
        como integrador."""
        return self.n_polos_reais + (1 if self.tem_polo_origem else 0)


# =============================================================================
# Modelo de Vector Fitting: ajusta, converte para TF, expõe diagnósticos
# =============================================================================

class VectorFittingModel:
    """
    Ajusta um modelo de função de transferência a dados de resposta em
    frequência via Vector Fitting (scikit-rf), e converte o resultado
    (polos + resíduos) para a forma polinomial Num(s)/Den(s) usada pelo
    restante do pipeline (python-control, tuning_analyzer.py).

    Não conhece arquivos, plots ou exportação — só a matemática do ajuste.
    """

    def __init__(self, config: VectorFittingConfig):
        self.config = config
        self.vf: Optional[VectorFitting] = None
        self.network: Optional[skrf.Network] = None

        # Preenchidos após fit()
        self.poles_half: np.ndarray = np.array([])   # só metade dos complexos (convenção skrf)
        self.residues_half: np.ndarray = np.array([])
        self.d: Optional[float] = None
        self.e: Optional[float] = None
        self.rms_error: Optional[float] = None

    # ---------- API pública ----------

    def fit(self, frequency_hz: np.ndarray, frdata: np.ndarray) -> "VectorFittingModel":
        """Ajusta o modelo aos dados (frequency_hz em Hz, frdata = resposta
        complexa medida). Retorna self para permitir encadeamento."""
        freq_skrf = skrf.Frequency.from_f(frequency_hz, unit="hz")
        s_data = frdata.reshape(-1, 1, 1)  # Network 1-porta fictícia
        self.network = skrf.Network(frequency=freq_skrf, s=s_data, name="bode_data")

        self.vf = VectorFitting(self.network)
        self.vf.max_iterations = self.config.max_iteracoes
        self.vf.max_tol = self.config.tolerancia

        self.vf.vector_fit(
            n_poles_real=self.config.n_polos_reais_total,
            n_poles_cmplx=self.config.n_pares_polos_complexos,
        )

        logger.info("Vector Fitting concluído.")
        logger.info("Tempo de execução: %.4f s", self.vf.wall_clock_time)

        self.rms_error = self.vf.get_rms_error()
        logger.info("Erro RMS do ajuste: %.6e", self.rms_error)

        if self.config.forcar_passividade:
            if not self.vf.is_passive():
                logger.info("Modelo não-passivo detectado. Aplicando enforcement de passividade...")
                self.vf.passivity_enforce()
            else:
                logger.info("Modelo já é passivo.")

        self.poles_half = self.vf.poles
        self.residues_half = self.vf.residues[0, :]
        self.d = self.vf.constant_coeff[0]
        self.e = self.vf.proportional_coeff[0]

        return self

    def to_tf(self) -> ct.TransferFunction:
        """Converte o modelo polo-resíduo ajustado para a forma polinomial
        Num(s)/Den(s), reconstruindo o par conjugado de cada polo complexo
        antes da conversão (o scikit-rf armazena só metade de cada par)."""
        self._check_fitted()

        poles_full, residues_full = self._full_conjugate_pairs()
        k_poly = [self.e, self.d]

        num_poly, den_poly = sig.invres(residues_full, poles_full, k_poly, tol=1e-8, rtype="avg")
        num_poly = np.real(num_poly)
        den_poly = np.real(den_poly)

        return ct.tf(num_poly, den_poly)

    def to_json_dict(self) -> dict:
        """Coeficientes Num/Den da TF ajustada, prontos para exportação em JSON."""
        sys_tf = self.to_tf()
        return {
            "num": [float(c) for c in sys_tf.num[0][0]],
            "den": [float(c) for c in sys_tf.den[0][0]],
            "dt": sys_tf.dt if sys_tf.dt is not None else 0,
            "rms_error": float(self.rms_error),
        }

    def summary(self) -> str:
        """Texto legível com polos, resíduos e diagnósticos do ajuste."""
        self._check_fitted()
        lines = ["=== POLOS E RESÍDUOS ESTIMADOS (Vector Fitting) ==="]
        for i, (p, r) in enumerate(zip(self.poles_half, self.residues_half), 1):
            tipo = "Complexo" if (np.iscomplex(p) and p.imag != 0) else "Real"
            lines.append(f"Polo {i} ({tipo}): p = {p:.4f} rad/s  |  resíduo = {r:.4f}")

        lines.append(f"\nTermo constante d = {self.d:.6f}")
        lines.append(f"Termo proporcional e (coef. de s) = {self.e:.6e}")

        sys_tf = self.to_tf()
        lines.append("\n=== OBJETO GERADO (python-control) ===")
        lines.append(str(sys_tf))

        lines.append("\n========== ZEROS (raiz bruta) ==========")
        for z in ct.zeros(sys_tf):
            lines.append(str(z))

        lines.append("\n========== POLOS (raiz bruta) ==========")
        for p in ct.poles(sys_tf):
            lines.append(str(p))

        lines.append("\n=== ZEROS EM wn / zeta (para comparar com o ajuste paramétrico) ===")
        lines.extend(self._roots_as_wn_zeta_lines(ct.zeros(sys_tf)))

        lines.append("\n=== POLOS EM wn / zeta (para comparar com o ajuste paramétrico) ===")
        lines.extend(self._roots_as_wn_zeta_lines(ct.poles(sys_tf)))

        return "\n".join(lines)

    # ---------- Métodos internos ----------

    def _check_fitted(self) -> None:
        if self.vf is None:
            raise RuntimeError("O modelo ainda não foi ajustado. Chame fit() primeiro.")

    @staticmethod
    def _root_to_wn_zeta(root: complex) -> Tuple[float, Optional[float]]:
        """Converte uma raiz (polo ou zero) da forma s = -zeta*wn ± j*wn*sqrt(1-zeta^2)
        para (wn, zeta). Para raiz puramente real, zeta não é definido (retorna None) --
        raiz real corresponde a um termo de 1a ordem (1/wn), não a wn/zeta de 2a ordem."""
        wn = abs(root)
        if wn == 0:
            return 0.0, None
        if root.imag == 0:
            return wn, None  # raiz real: termo de 1a ordem, sem zeta
        zeta = -root.real / wn
        return wn, zeta

    @classmethod
    def _roots_as_wn_zeta_lines(cls, roots) -> List[str]:
        """Agrupa raízes conjugadas em pares e formata como wn/zeta; raízes
        reais (incluindo a raiz na origem) são listadas separadamente."""
        lines: List[str] = []
        vistos = set()

        for i, root in enumerate(roots):
            if i in vistos:
                continue

            wn, zeta = cls._root_to_wn_zeta(complex(root))

            if zeta is None:
                if wn == 0:
                    lines.append("Raiz na origem (integrador/diferenciador)")
                else:
                    lines.append(f"Raiz real: 1/wn -> wn = {wn:.4f} rad/s")
                continue

            # Marca o conjugado correspondente como já visto, se existir
            for j, outro in enumerate(roots):
                if j != i and j not in vistos and abs(complex(outro) - complex(root).conjugate()) < 1e-6:
                    vistos.add(j)
                    break

            lines.append(f"Par complexo: wn = {wn:.4f} rad/s | zeta = {zeta:.4f}")

        return lines

    def _full_conjugate_pairs(self) -> Tuple[np.ndarray, np.ndarray]:
        """Reconstrói o conjunto completo de polos/resíduos, adicionando o
        par conjugado de cada polo complexo (o scikit-rf só armazena a metade
        com parte imaginária positiva)."""
        poles_full, residues_full = [], []
        for p, r in zip(self.poles_half, self.residues_half):
            poles_full.append(p)
            residues_full.append(r)
            if np.iscomplex(p) and p.imag != 0:
                poles_full.append(np.conj(p))
                residues_full.append(np.conj(r))
        return np.array(poles_full), np.array(residues_full)


@dataclass
class SweepResult:
    """Resultado de uma combinação testada na varredura de ordem do modelo."""
    n_polos_reais: int
    n_pares_polos_complexos: int
    tem_polo_origem: bool
    rms_error: float
    n_polos_total: int  # graus de liberdade totais -- útil para julgar parcimônia

    def format_line(self) -> str:
        return (
            f"polos_reais={self.n_polos_reais} | pares_complexos={self.n_pares_polos_complexos} | "
            f"integrador={self.tem_polo_origem} | n_polos_total={self.n_polos_total} | "
            f"RMS={self.rms_error:.6e}"
        )


# =============================================================================
# Orquestração: carrega o Bode, ajusta o modelo, plota, exporta
# =============================================================================

class VectorFittingIdentifier:
    """
    Orquestra a identificação de função de transferência via Vector Fitting a
    partir de um único arquivo de Bode Plot: carrega os dados, ajusta um
    VectorFittingModel, gera os plots de diagnóstico e exporta o resultado
    em JSON.

    Uso:
        identifier = VectorFittingIdentifier(
            bode_file="./bode_files_PAPU/Id_1_Pos_kp_10_Tn_0_20.csv",
            vf_config=VectorFittingConfig(n_polos_reais=2, n_pares_polos_complexos=1),
            system_part="Open-Loop",
        )
        identifier.run(output_dir="./tfs_json_PAPU", plot_dir="./bodes_pngs")
    """

    def __init__(
        self,
        bode_file: Union[str, Path],
        vf_config: VectorFittingConfig,
        system_part: str = "Open-Loop",
    ):
        self.bode_file = Path(bode_file)
        self.system_part = system_part
        self.model = VectorFittingModel(vf_config)

        self.sys_frd: Optional[ct.FrequencyResponseData] = None
        self.frequency_hz: Optional[np.ndarray] = None
        self.frdata: Optional[np.ndarray] = None

    # ---------- API pública ----------

    def run(self, output_dir: Union[str, Path], plot_dir: Union[str, Path]) -> Path:
        """Executa o pipeline completo. Retorna o caminho do JSON exportado."""
        self._load_bode()
        self.model.fit(self.frequency_hz, self.frdata)
        logger.info("\n%s", self.model.summary())

        json_path = self._export_json(Path(output_dir))
        self._plot_all(Path(plot_dir))
        return json_path

    def sweep(
        self,
        n_polos_reais_range: List[int],
        n_pares_polos_complexos_range: List[int],
        tem_polo_origem_options: Optional[List[bool]] = None,
    ) -> List[SweepResult]:
        """Testa uma grade (grid search) de combinações de ordem do modelo e
        reporta o erro RMS de cada uma, sem gerar plots nem exportar JSON --
        serve para decidir a ordem antes de rodar run().

        Como o espaço de busca é pequeno e discreto (poucos polos, valores
        inteiros), uma varredura exaustiva é mais simples e mais
        interpretável que uma busca bayesiana (tipo Optuna): dá pra ver a
        tabela completa em vez de confiar numa heurística de amostragem.

        Retorna a lista de resultados, ordenada do menor para o maior erro RMS.
        """
        if self.frequency_hz is None:
            self._load_bode()

        tem_polo_origem_options = (
            tem_polo_origem_options if tem_polo_origem_options is not None
            else [self.model.config.tem_polo_origem]
        )

        resultados: List[SweepResult] = []
        combinacoes = list(itertools.product(
            n_polos_reais_range, n_pares_polos_complexos_range, tem_polo_origem_options,
        ))

        logger.info("Testando %d combinações de ordem do modelo...", len(combinacoes))

        for n_reais, n_pares_complexos, tem_origem in combinacoes:
            if n_reais == 0 and n_pares_complexos == 0 and not tem_origem:
                continue  # modelo vazio, sem nenhum polo -- não faz sentido

            config = VectorFittingConfig(
                n_polos_reais=n_reais,
                n_pares_polos_complexos=n_pares_complexos,
                tem_polo_origem=tem_origem,
                max_iteracoes=self.model.config.max_iteracoes,
                tolerancia=self.model.config.tolerancia,
                forcar_passividade=False,
            )
            model = VectorFittingModel(config)

            try:
                model.fit(self.frequency_hz, self.frdata)
                n_polos_total = config.n_polos_reais_total + 2 * n_pares_complexos
                resultados.append(SweepResult(
                    n_polos_reais=n_reais, n_pares_polos_complexos=n_pares_complexos,
                    tem_polo_origem=tem_origem, rms_error=model.rms_error,
                    n_polos_total=n_polos_total,
                ))
            except Exception as e:
                logger.warning(
                    "Falha ao ajustar com polos_reais=%d, pares_complexos=%d, integrador=%s: %s",
                    n_reais, n_pares_complexos, tem_origem, e,
                )

        resultados.sort(key=lambda r: r.rms_error)

        logger.info("\n=== RESULTADO DA VARREDURA (ordenado por erro RMS) ===")
        for r in resultados:
            logger.info(r.format_line())

        return resultados

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
        self.frequency_hz = df["Frequency"].to_numpy()

        mag_linear = 10 ** (gain / 20)
        phase_rad = np.deg2rad(phase)
        omega = 2 * np.pi * self.frequency_hz

        self.frdata = mag_linear * np.exp(1j * phase_rad)
        self.sys_frd = ct.frd(self.frdata, omega, name=f"{self.system_part} Bode Data")

    def _base_output_name(self) -> str:
        return f"VF_{self.system_part.lower()}_TF_{self.bode_file.stem}"

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

        self._plot_dominio_dados(base_path, sys_tf)
        omega_ext = self._plot_dominio_expandido(base_path, sys_tf)
        self._plot_nativo_skrf(base_path)
        self._plot_com_margens(base_path, sys_tf, omega_ext)

    def _plot_dominio_dados(self, base_path: Path, sys_tf) -> None:
        ct.bode_plot(
            [self.sys_frd, sys_tf], omega=self.sys_frd.omega, dB=True, Hz=True,
            legend_loc="lower left",
        )
        plt.suptitle("Comparative Bode Plot")
        out = f"{base_path}_DOMINIO_DADOS.png"
        plt.savefig(out)
        logger.info("Saved %s", out)
        plt.close()

    def _plot_dominio_expandido(self, base_path: Path, sys_tf) -> np.ndarray:
        omega_dados = self.sys_frd.omega
        omega_min_ext = omega_dados.min() / 1000
        omega_max_ext = omega_dados.max() * 10
        omega_ext = np.logspace(np.log10(omega_min_ext), np.log10(omega_max_ext), 500)

        resp_tf = ct.frequency_response(sys_tf, omega_ext)
        mag_tf_db = 20 * np.log10(np.abs(resp_tf.frdata[0, 0]))
        fase_tf_deg = np.rad2deg(np.unwrap(np.angle(resp_tf.frdata[0, 0])))

        fig, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

        ax_mag.semilogx(omega_dados, 20 * np.log10(np.abs(self.frdata)), "o", label="Dados (FRD)", markersize=4)
        ax_mag.semilogx(omega_ext, mag_tf_db, label="Modelo (Vector Fitting)")
        ax_mag.set_ylabel("Magnitude (dB)")
        ax_mag.legend()
        ax_mag.grid(True, which="both")

        ax_phase.semilogx(omega_dados, np.rad2deg(np.unwrap(np.angle(self.frdata))), "o", markersize=4)
        ax_phase.semilogx(omega_ext, fase_tf_deg)
        ax_phase.set_ylabel("Fase (graus)")
        ax_phase.set_xlabel("Frequência (rad/s)")
        ax_phase.grid(True, which="both")

        pad_mag, pad_fase = 5, 15
        ax_mag.set_ylim(mag_tf_db.min() - pad_mag, mag_tf_db.max() + pad_mag)
        ax_phase.set_ylim(fase_tf_deg.min() - pad_fase, fase_tf_deg.max() + pad_fase)

        plt.tight_layout()
        out = f"{base_path}_DOMINIO_EXPANDIDO.png"
        plt.savefig(out)
        logger.info("Saved %s", out)
        plt.close()

        return omega_ext

    def _plot_nativo_skrf(self, base_path: Path) -> None:
        fig_vf, ax_vf = plt.subplots(figsize=(8, 4))
        self.model.vf.plot_s_db(0, 0, ax=ax_vf)
        ax_vf.set_title("Vector Fitting: Magnitude (dB) - dados vs. modelo")
        plt.tight_layout()
        out = f"{base_path}_SKRF_NATIVO.png"
        plt.savefig(out)
        logger.info("Saved %s", out)
        plt.close()

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
        description="Identifica a função de transferência de um sistema via "
        "Vector Fitting, a partir de um único arquivo de Bode Plot experimental."
    )
    parser.add_argument("--bode-file", type=str, required=True, help="Arquivo .csv do Bode a analisar.")
    parser.add_argument(
        "--vf-config", type=str, default=None,
        help="Arquivo JSON com a configuração do Vector Fitting (ordem do modelo, "
        "iterações, tolerância). Se omitido, usa os valores padrão.",
    )
    parser.add_argument(
        "--system-part", type=str, default="Open-Loop",
        choices=["Process", "Open-Loop", "Close-Loop"],
        help="Parte do sistema a identificar.",
    )
    parser.add_argument("--output-dir", type=str, default="./tfs_json_PAPU", help="Pasta de saída do JSON.")
    parser.add_argument("--plot-dir", type=str, default="./bodes_pngs", help="Pasta de saída dos gráficos.")
    parser.add_argument("--show", action="store_true", help="Exibe os gráficos na tela além de salvar.")
    parser.add_argument(
        "--sweep", action="store_true",
        help="Em vez de ajustar e exportar, testa uma grade de ordens de modelo "
        "e reporta o erro RMS de cada uma (não gera plots nem JSON).",
    )
    parser.add_argument(
        "--sweep-polos-reais", type=int, nargs="+", default=[0, 1, 2, 3],
        help="Valores de n_polos_reais a testar no --sweep.",
    )
    parser.add_argument(
        "--sweep-pares-complexos", type=int, nargs="+", default=[0, 1, 2],
        help="Valores de n_pares_polos_complexos a testar no --sweep.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    config = (
        VectorFittingConfig.from_json(args.vf_config)
        if args.vf_config else VectorFittingConfig()
    )
    identifier = VectorFittingIdentifier(
        bode_file=args.bode_file, vf_config=config, system_part=args.system_part,
    )

    if args.sweep:
        identifier.sweep(
            n_polos_reais_range=args.sweep_polos_reais,
            n_pares_polos_complexos_range=args.sweep_pares_complexos,
        )
        return

    identifier.run(output_dir=args.output_dir, plot_dir=args.plot_dir)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()  