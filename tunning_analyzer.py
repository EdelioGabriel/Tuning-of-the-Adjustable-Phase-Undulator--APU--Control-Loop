"""
Análise estatística de um lote de funções de transferência identificadas
(exportadas em JSON pelo frf_identifier.py) e cálculo do ajuste de ganho (K)
necessário para atingir uma margem de ganho alvo — com decisão automática
entre um ajuste único (modelos consistentes) ou três cenários
(pessimista/médio/otimista) quando os modelos divergem muito entre si.

Autor original: Edélio Gabriel Magalhães de Jesus
Refatorado para estrutura orientada a objetos com auxílio de IA

Uso:

python tunning_analyzer.py --data-dir "./tfs_json_PAPU" --otimizador "LS" --system-part "OPEN-LOOP" --variavel "Vel" --alvo-mg 10 --limite-desvio 0.10 --output-dir "./tunning_results"
"""

import argparse
import glob
import json
import logging
import os                                                           
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Union

import control as ct
import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


@dataclass
class MarginStats:
    """Estatísticas de margem de ganho/fase calculadas sobre um lote de TFs."""
    gains_db: List[float]
    phases_deg: List[float]
    media_mg: float
    desvio_mg: float
    media_mf: float
    desvio_mf: float

    def coef_variacao_aceitavel(self, limite_pct: float) -> bool:
        """True se o desvio padrão, relativo à média, estiver dentro do limite
        para MG e MF (coeficiente de variação)."""
        return (
            abs(self.desvio_mg / self.media_mg) < limite_pct
            and abs(self.desvio_mf / self.media_mf) < limite_pct
        )


@dataclass
class TuningScenario:
    """Um cenário de ajuste: um valor de K aplicado a um sistema de referência."""
    nome: str
    k_adj: float
    mg_base_db: float
    sys_final: ct.TransferFunction
    gm_final_db: float
    pm_final_deg: float


class TuningAnalyzer:
    """
    Carrega um lote de funções de transferência identificadas (JSON), calcula
    margens de ganho/fase, e decide o(s) ajuste(s) de ganho necessário(s) para
    atingir uma margem de ganho alvo.

    Se os modelos forem estatisticamente consistentes entre si (baixo desvio),
    gera um único ajuste baseado na média. Se divergirem, gera três cenários
    (pessimista, médio, otimista) baseados no pior, na média, e no melhor caso.

    Uso:
        analyzer = TuningAnalyzer(
            data_dir="./tfs_json_PAPU",
            otimizador="LS",
            parte_do_sistema="open-loop",
            variavel="Vel",
            alvo_mg_db=10.0,
            limite_desvio_pct=0.10,
        )
        analyzer.run(output_dir="./tunning_results")
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        otimizador: str = "LS",
        parte_do_sistema: str = "open-loop",
        variavel: str = "",
        alvo_mg_db: float = 10.0,
        limite_desvio_pct: float = 0.10,    
        omega_vetor: Optional[np.ndarray] = None,
    ):
        self.data_dir = Path(data_dir)
        self.otimizador = otimizador
        self.parte_do_sistema = parte_do_sistema
        self.variavel = variavel
        self.alvo_mg_db = alvo_mg_db
        self.limite_desvio_pct = limite_desvio_pct
        self.omega_vetor = omega_vetor if omega_vetor is not None else np.logspace(-3, 4, 1500)

        self.sys_tfs: List[ct.TransferFunction] = []
        self.file_paths: List[Path] = []
        self.stats: Optional[MarginStats] = None
        self.scenarios: List[TuningScenario] = []

    # ---------- API pública ----------

    def run(self, output_dir: Union[str, Path]) -> List[TuningScenario]:
        """Executa o pipeline completo: carregar, calcular margens, decidir
        cenário(s), plotar e salvar. Retorna a lista de cenários gerados."""
        self._load_all()

        if not self.sys_tfs:
            raise ValueError(
                f"Nenhuma TF encontrada em '{self.data_dir}' com o padrão "
                f"'{self._file_pattern()}'."
            )

        self.stats = self._compute_stats()
        self._log_report()

        output_dir = Path(output_dir)
        base_path = output_dir / f"TUNNING_{self.otimizador}s_{self.parte_do_sistema.lower()}_TFs"

        if self.stats.coef_variacao_aceitavel(self.limite_desvio_pct):
            logger.info("Validação Estatística Sucedida: Baixa variação entre os modelos.")
            self.scenarios = [self._build_scenario("Médio (único)", self.stats.media_mg)]
        else:
            logger.info("Validação Estatística Recusada: Modelos divergem muito entre si.")
            pior_mg = np.nanmin(self.stats.gains_db)
            melhor_mg = np.nanmax(self.stats.gains_db)
            self.scenarios = [
                self._build_scenario("Pessimista (pior caso)", pior_mg),
                self._build_scenario("Médio", self.stats.media_mg),
                self._build_scenario("Otimista (melhor caso)", melhor_mg),
            ]

        self._log_scenarios()
        self._plot_scenarios(output_dir, base_path)
        self._plot_all_models(output_dir, base_path)

        return self.scenarios

    # ---------- Métodos internos ----------

    def _file_pattern(self) -> str:
        return f"{self.otimizador}*{self.parte_do_sistema}*{self.variavel}*.json"

    @staticmethod
    def _load_tf(path: Path) -> ct.TransferFunction:
        with open(path, "r") as f:
            dados = json.load(f)
        sys_tf = ct.tf(dados["num"], dados["den"])
        return ct.minreal(sys_tf, verbose=False)

    def _load_all(self) -> None:
        pattern = os.path.join(str(self.data_dir), self._file_pattern())
        self.file_paths = [Path(p) for p in sorted(glob.glob(pattern))]
        self.sys_tfs = [self._load_tf(p) for p in self.file_paths]

    def _compute_stats(self) -> MarginStats:
        gains_db, phases_deg = [], []
        for tf in self.sys_tfs:
            gm, pm, _, _ = ct.margin(tf)
            gm_db = 20 * np.log10(gm) if gm > 0 else np.nan
            gains_db.append(gm_db)
            phases_deg.append(pm)

        return MarginStats(
            gains_db=gains_db,
            phases_deg=phases_deg,
            media_mg=float(np.nanmean(gains_db)),
            desvio_mg=float(np.nanstd(gains_db)),
            media_mf=float(np.nanmean(phases_deg)),
            desvio_mf=float(np.nanstd(phases_deg)),
        )

    def _log_report(self) -> None:
        logger.info("--- RELATÓRIO DE MARGENS INDIVIDUAIS ---")
        for i, (path, gm_db, pm) in enumerate(
            zip(self.file_paths, self.stats.gains_db, self.stats.phases_deg), 1
        ):
            logger.info("Modelo %d (%s): MG = %.2f dB | MF = %.2f°", i, path.name, gm_db, pm)

        logger.info("\n--- ANÁLISE ESTATÍSTICA ---")
        logger.info(
            "Média das Margens  -> MG: %.2f dB | MF: %.2f°",
            self.stats.media_mg, self.stats.media_mf,
        )
        logger.info(
            "Desvio Padrão      -> MG: %.2f dB | MF: %.2f°",
            self.stats.desvio_mg, self.stats.desvio_mf,
        )
        logger.info("\n--- PROCESSO DE TUNING ---")

    def _build_scenario(self, nome: str, mg_base_db: float) -> TuningScenario:
        delta_db = mg_base_db - self.alvo_mg_db
        k_adj = 10 ** (delta_db / 20)

        sys_final = self.sys_tfs[0] * k_adj
        gm_f, pm_f, _, _ = ct.margin(sys_final)
        gm_f_db = 20 * np.log10(gm_f) if gm_f > 0 else np.nan

        return TuningScenario(
            nome=nome, k_adj=k_adj, mg_base_db=mg_base_db,
            sys_final=sys_final, gm_final_db=gm_f_db, pm_final_deg=pm_f,
        )

    def _log_scenarios(self) -> None:
        for s in self.scenarios:
            logger.info(
                "   %s (MG base %.1f dB): K_adj = %.3f -> Nova MG: %.2f dB | Nova MF: %.2f°",
                s.nome, s.mg_base_db, s.k_adj, s.gm_final_db, s.pm_final_deg,
            )

    def _plot_scenarios(self, output_dir: Path, base_path: Path) -> None:
        output_dir.mkdir(parents=True, exist_ok=True)

        for s in self.scenarios:
            ct.bode_plot(
                s.sys_final, omega=self.omega_vetor, dB=True, Hz=False,
                label=f"Modelo {s.nome}", display_margins=True,
            )
            plt.suptitle(f"Comparative Bode Plot - {s.nome}")
            safe_nome = s.nome.split(" ")[0]
            out = f"{base_path}_DOMINIO_DADOS_{safe_nome}.png"
            plt.savefig(out)
            logger.info("Saved %s", out)
            plt.close()

    def _plot_all_models(self, output_dir: Path, base_path: Path) -> None:
        plt.figure(figsize=(10, 5))
        for i, tf in enumerate(self.sys_tfs, 1):
            ct.bode_plot(
                tf, omega=self.omega_vetor, dB=True, Hz=False,
                label=f"Modelo {i}", display_margins=True,
            )
        plt.legend()
        plt.suptitle("Comparação de Resposta de Frequência dos Modelos Carregados")
        out = f"{base_path}_TODOS_MODELOS.png"
        plt.savefig(out)
        logger.info("Saved %s", out)
        plt.close()


def _parse_args() -> "argparse.Namespace":
    parser = argparse.ArgumentParser(
        description="Analisa um lote de funções de transferência identificadas "
        "e calcula o(s) ajuste(s) de ganho necessário(s)."
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Pasta com os JSONs de TF.")
    parser.add_argument("--otimizador", type=str, default="LS", help="Prefixo do otimizador no nome do arquivo (ex: 'LS').")
    parser.add_argument("--system-part", type=str, default="open-loop", help="Parte do sistema no nome do arquivo.")
    parser.add_argument("--variavel", type=str, default="", help="Variável no nome do arquivo (ex: 'Vel', 'Pos').")
    parser.add_argument("--alvo-mg", type=float, default=10.0, help="Margem de ganho alvo, em dB.")
    parser.add_argument("--limite-desvio", type=float, default=0.10, help="Limite de coeficiente de variação (fração, ex: 0.10 = 10%%).")
    parser.add_argument("--output-dir", type=str, default="./tunning_results", help="Pasta de saída dos resultados.")
    parser.add_argument("--show", action="store_true", help="Exibe os gráficos na tela além de salvar.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    analyzer = TuningAnalyzer(
        data_dir=args.data_dir,
        otimizador=args.otimizador,
        parte_do_sistema=args.system_part,
        variavel=args.variavel,
        alvo_mg_db=args.alvo_mg,
        limite_desvio_pct=args.limite_desvio,
    )
    analyzer.run(output_dir=args.output_dir)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()