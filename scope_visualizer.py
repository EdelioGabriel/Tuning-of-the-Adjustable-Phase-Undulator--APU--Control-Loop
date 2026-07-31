"""
Leitura e plotagem de arquivos .csv exportados pelo ScopeWizard (TwinCAT),
contendo respostas no tempo (ex: resposta a um degrau), com múltiplas
variáveis dispostas lado a lado (cada uma com seu par de colunas
índice/tempo e valor), separadas por tab, aceitando vírgula ou ponto como
separador decimal.

Autor original: Edélio Gabriel Magalhães de Jesus
Refatorado para estrutura orientada a objetos com auxílio de IA.

Uso:

python scope_visualizer.py --data-dir "./scope_view_files" --var-filter "Velo" --setpoint "Velo=20.0" --output-dir "./scope_view_pngs" --regime-delay 2 --regime-window 10

ou 

python scope_visualizer.py --data-dir "./scope_view_files" --var-filter "Velo" --var-filter "Pos" --setpoint "Velo=10.0" --setpoint "Pos=5.0" --output-dir "./scope_view_pngs"
"""

import argparse
import glob
import io
import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


@dataclass
class StepResponseMetrics:
    """Métricas de resposta a um degrau para uma variável, dado um setpoint."""
    variavel: str
    setpoint: float
    valor_pico: float
    valor_final: float
    overshoot_pct: float
    erro_regime_pct: float
    # Erro RMS (raiz quadrada média) do sinal de erro e(t) = setpoint - Act*(t)
    # dentro da janela de regime permanente, em unidades absolutas e em %
    # do setpoint. Métrica mais adequada que o erro médio quando o sinal
    # oscila/tem ruído em regime, pois capta a "energia" do erro.
    erro_rms: float = 0.0
    erro_rms_pct: float = 0.0
    # Amplitude da oscilação em regime permanente (máximo e mínimo do sinal
    # dentro da janela de regime), usados para desenhar a banda de
    # tolerância no gráfico de erro.
    banda_max: float = 0.0
    banda_min: float = 0.0
    # índice (posição na série) a partir do qual a janela de regime
    # permanente foi considerada, útil para depuração/plot.
    regime_start_idx: int = 0
    regime_end_idx: int = 0

    def format_line(self) -> str:
        return (
            f"{self.variavel}: overshoot = {self.overshoot_pct:.1f}% | "
            f"erro RMS = {self.erro_rms_pct:.1f}%"
        )

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class ScopeVisualizer:
    """
    Carrega arquivos .csv exportados pelo ScopeWizard (TwinCAT) de uma pasta,
    contendo respostas no tempo com múltiplas variáveis lado a lado, e plota
    (um PNG por arquivo) as variáveis cujo nome contenha algum dos filtros
    indicados (ex: 'Velo', 'Pos').

    Uso:
        viz = ScopeVisualizer(data_dir="./scope_view_files", var_filters=["Velo"])
        viz.run(output_dir="./scope_view_pngs")
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        var_filters: Optional[List[str]] = None,
        y_label: str = "Velocidade (mm/s)",
        setpoints: Optional[Dict[str, float]] = None,
        regime_delay_s: float = 0.3,
        regime_window_s: float = 0.2,
    ):
        self.data_dir = Path(data_dir)
        self.var_filters = var_filters if var_filters else ["Velo"]
        self.y_label = y_label
        # setpoints: mapa {substring_da_variavel: valor_setpoint}. Uma variável
        # usa o setpoint cuja chave aparece no seu nome (ex: {"Velo": 10.0}).
        self.setpoints = setpoints or {}
        # tempo (em segundos), após o instante do pico (overshoot), que é
        # aguardado antes de começar a medir o regime permanente. Evita
        # medir durante o transiente logo após o pico.
        self.regime_delay_s = regime_delay_s
        # duração (em segundos) da janela usada para calcular o valor final
        # (regime permanente), a partir do fim do atraso pós-pico.
        self.regime_window_s = regime_window_s

        self.dfs: List[pd.DataFrame] = []
        self.labels: List[str] = []
        self.metrics: Dict[str, List[StepResponseMetrics]] = {}

    # ---------- API pública ----------

    def run(self, output_dir: Union[str, Path]) -> List[Path]:
        """Executa o pipeline completo: carregar arquivos e plotar cada um.

        Retorna a lista de caminhos dos PNGs salvos."""
        self._load_all()

        if not self.dfs:
            raise ValueError(f"Nenhum arquivo foi carregado com sucesso a partir de '{self.data_dir}'.")

        return self._plot_all(Path(output_dir))

    # ---------- Métodos internos ----------

    @staticmethod
    def _eh_numero(campo: str) -> bool:
        """True se o campo (string) representa um número, com decimal ',' ou '.'."""
        s = campo.strip().lstrip("-").replace(",", ".", 1)
        return s.replace(".", "", 1).isdigit()

    def _load_scope_csv(self, arquivo: Path) -> pd.DataFrame:
        with open(arquivo, "r", encoding="utf-8-sig") as f:
            linhas = [l.rstrip("\n") for l in f]
        campos_por_linha = [l.split("\t") for l in linhas]

        idx_header = next(
            i for i, c in enumerate(campos_por_linha)
            if len(c) >= 4 and c[0] == "Name" and c[2] == "Name"
        )
        nomes_variaveis = [c.strip() for c in campos_por_linha[idx_header][1::2] if c.strip()]

        idx_dados = next(
            i for i in range(idx_header + 1, len(campos_por_linha))
            if self._eh_numero(campos_por_linha[i][0])
        )

        texto_dados = "\n".join(l for l in linhas[idx_dados:] if l.strip())
        decimal_sep = "," if "," in texto_dados.split("\n", 1)[0] else "."

        df_raw = pd.read_csv(
            io.StringIO(texto_dados), sep="\t", header=None,
            decimal=decimal_sep, engine="python",
        )

        df = pd.DataFrame({"Time_ms": df_raw[0].astype(float)})
        for j, nome in enumerate(nomes_variaveis):
            df[nome] = df_raw[2 * j + 1].astype(float)

        return df

    def _load_all(self) -> None:
        self.dfs = []
        self.labels = []

        arquivos_csv = sorted(self.data_dir.glob("*.csv"))

        for arquivo in arquivos_csv:
            nome_experimento = arquivo.stem
            try:
                df = self._load_scope_csv(arquivo)
                self.dfs.append(df)
                self.labels.append(nome_experimento)
                logger.info(
                    "%s: %d amostras carregadas (%s)",
                    nome_experimento, len(df), list(df.columns[1:]),
                )
            except Exception as e:
                logger.warning("Erro ao processar o arquivo %s: %s", nome_experimento, e)

    def _matching_columns(self, df: pd.DataFrame) -> List[str]:
        return [c for c in df.columns if any(f in c for f in self.var_filters)]

    def _setpoint_for(self, var_name: str) -> Optional[float]:
        """Retorna o setpoint configurado cuja chave aparece no nome da
        variável (ex: setpoints={'Velo': 10.0} casa com 'Velo_Feedback')."""
        for chave, valor in self.setpoints.items():
            if chave in var_name:
                return valor
        return None

    @staticmethod
    def _is_setpoint_column(var_name: str) -> bool:
        """True se a variável é a própria coluna de setpoint (ex: 'SetVelo'),
        e não uma variável ativa (ex: 'ActVelo'). Colunas de setpoint não
        recebem métricas, pois são a referência, não a resposta medida."""
        return "Set" in var_name

    def _regime_window_after_peak(
        self, tempo_s: np.ndarray, serie: np.ndarray, idx_pico: int,
    ) -> "tuple[int, int]":
        """Determina a janela de regime permanente a partir de um atraso
        fixo (regime_delay_s) após o instante do pico (overshoot), com
        duração fixa (regime_window_s). Retorna (inicio_idx, fim_idx) da
        janela (fim_idx exclusivo).

        Se a janela ultrapassar o fim da série, é truncada até o último
        ponto disponível.
        """
        t_pico = tempo_s[idx_pico]
        t_inicio_janela = t_pico + self.regime_delay_s
        t_fim_janela = t_inicio_janela + self.regime_window_s

        inicio_idx = int(np.searchsorted(tempo_s, t_inicio_janela, side="left"))
        fim_idx = int(np.searchsorted(tempo_s, t_fim_janela, side="right"))

        n_total = len(serie)
        inicio_idx = min(inicio_idx, n_total - 1)
        fim_idx = min(max(fim_idx, inicio_idx + 1), n_total)

        n_amostras_janela = fim_idx - inicio_idx
        if n_amostras_janela < 5:
            logger.warning(
                "A janela de regime permanente ficou com apenas %d amostra(s) "
                "(dados insuficientes após o pico + atraso de %.2fs). O erro de "
                "regime calculado pode não ser confiável; considere reduzir "
                "--regime-delay/--regime-window ou verificar se o arquivo cobre "
                "tempo suficiente após o pico.",
                n_amostras_janela, self.regime_delay_s,
            )

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[DEBUG regime] idx_pico=%d t_pico=%.4f t_inicio_janela=%.4f t_fim_janela=%.4f "
                "inicio_idx=%d fim_idx=%d janela_valores=%s",
                idx_pico, t_pico, t_inicio_janela, t_fim_janela, inicio_idx, fim_idx,
                serie[inicio_idx:fim_idx].tolist(),
            )

        return inicio_idx, fim_idx

    def _compute_metrics(
        self, df: pd.DataFrame, var_name: str, setpoint: float,
    ) -> StepResponseMetrics:
        serie = df[var_name].to_numpy()
        tempo_s = (df["Time_ms"].to_numpy()) / 1000.0

        # Pico na direção do degrau (funciona para setpoint positivo ou negativo).
        idx_pico = int(np.argmax(serie)) if setpoint >= 0 else int(np.argmin(serie))
        valor_pico = float(serie[idx_pico])

        inicio_idx, fim_idx = self._regime_window_after_peak(tempo_s, serie, idx_pico)
        janela_regime = serie[inicio_idx:fim_idx]
        valor_final = float(np.mean(janela_regime))

        overshoot_pct = (valor_pico - setpoint) / setpoint * 100.0
        overshoot_pct = max(overshoot_pct, 0.0)  # sem overshoot negativo (undershoot não conta aqui)

        erro_regime_pct = (setpoint - valor_final) / setpoint * 100.0

        # Erro RMS em regime permanente: e(t) = setpoint - Act*(t) dentro da
        # janela de regime. Mais robusto que o erro médio para sinais com
        # ripple/oscilação sustentada, pois não cancela desvios simétricos
        # em torno do setpoint — mede a energia do erro, não só o
        # deslocamento do centro da oscilação.
        erro_instantaneo = setpoint - janela_regime
        erro_rms = float(np.sqrt(np.mean(erro_instantaneo ** 2)))
        erro_rms_pct = erro_rms / abs(setpoint) * 100.0

        banda_max = float(np.max(janela_regime))
        banda_min = float(np.min(janela_regime))

        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "[DEBUG metrics] variavel=%s valor_final=%.6f valor_pico=%.6f "
                "overshoot_pct=%.4f erro_regime_pct=%.4f erro_rms=%.6f erro_rms_pct=%.4f "
                "banda=[%.4f, %.4f]",
                var_name, valor_final, valor_pico, overshoot_pct, erro_regime_pct,
                erro_rms, erro_rms_pct, banda_min, banda_max,
            )

        return StepResponseMetrics(
            variavel=var_name, setpoint=setpoint, valor_pico=valor_pico,
            valor_final=valor_final, overshoot_pct=overshoot_pct,
            erro_regime_pct=erro_regime_pct, erro_rms=erro_rms, erro_rms_pct=erro_rms_pct,
            banda_max=banda_max, banda_min=banda_min,
            regime_start_idx=inicio_idx, regime_end_idx=fim_idx,
        )

    def _metrics_for_label(self, df: pd.DataFrame, variaveis: List[str]) -> List[StepResponseMetrics]:
        resultados = []
        for var in variaveis:
            if self._is_setpoint_column(var):
                continue  # colunas de setpoint não recebem métricas
            setpoint = self._setpoint_for(var)
            if setpoint is None:
                continue
            resultados.append(self._compute_metrics(df, var, setpoint))
        return resultados

    @staticmethod
    def _draw_metrics_box(ax, metrics: List[StepResponseMetrics]) -> None:
        if not metrics:
            return
        texto = "\n".join(m.format_line() for m in metrics)
        ax.text(
            0.98, 0.02, texto,
            transform=ax.transAxes, fontsize=9,
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.85, edgecolor="gray"),
        )

    @staticmethod
    def _plot_error_band(ax, df: pd.DataFrame, metrics: List[StepResponseMetrics]) -> None:
        """Plota o erro e(t) = setpoint - Act*(t) ao longo do tempo, dentro
        da janela de regime permanente de cada variável, com linhas
        tracejadas horizontais marcando a amplitude (máximo e mínimo) do
        erro observado -- a "banda de tolerância" da oscilação estável.
        """
        tempo_s = (df["Time_ms"].to_numpy()) / 1000.0
        algo_plotado = False

        for m in metrics:
            serie = df[m.variavel].to_numpy()
            i0, i1 = m.regime_start_idx, m.regime_end_idx
            if i1 <= i0:
                continue

            t_janela = tempo_s[i0:i1]
            erro_t = m.setpoint - serie[i0:i1]

            linha, = ax.plot(t_janela, erro_t, label=f"erro {m.variavel}")
            cor = linha.get_color()

            erro_banda_max = m.setpoint - m.banda_min  # pico do erro (Act* mínimo)
            erro_banda_min = m.setpoint - m.banda_max  # vale do erro (Act* máximo)
            ax.axhline(erro_banda_max, linestyle="--", color=cor, alpha=0.6, linewidth=1)
            ax.axhline(erro_banda_min, linestyle="--", color=cor, alpha=0.6, linewidth=1)
            ax.text(
                t_janela[-1], erro_banda_max, f" {erro_banda_max:.2f}",
                fontsize=7, color=cor, va="bottom",
            )
            ax.text(
                t_janela[-1], erro_banda_min, f" {erro_banda_min:.2f}",
                fontsize=7, color=cor, va="top",
            )
            algo_plotado = True

        if not algo_plotado:
            return

        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Tempo (s)")
        ax.set_ylabel("Erro (setpoint - Act*)")
        ax.set_title("Erro em regime permanente, com banda de tolerância (máx/mín)")
        ax.legend(loc="best", fontsize=8)
        ax.grid(True)

    def _plot_all(self, output_dir: Path) -> List[Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        output_paths: List[Path] = []
        self.metrics = {}

        for df, label in zip(self.dfs, self.labels):
            variaveis_filtradas = self._matching_columns(df)
            if not variaveis_filtradas:
                logger.info(
                    "%s: nenhuma variável casando com %s encontrada, pulando.",
                    label, self.var_filters,
                )
                continue

            metrics = self._metrics_for_label(df, variaveis_filtradas)
            self.metrics[label] = metrics
            for m in metrics:
                logger.info("%s -> %s", label, m.format_line())

            tem_erro_para_plotar = any(m.regime_end_idx > m.regime_start_idx for m in metrics)

            if tem_erro_para_plotar:
                fig, (ax_sinal, ax_erro) = plt.subplots(
                    2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]},
                )
            else:
                fig, ax_sinal = plt.subplots(figsize=(10, 5))
                ax_erro = None

            tempo_s = df["Time_ms"] / 1000.0
            for var in variaveis_filtradas:
                ax_sinal.plot(tempo_s, df[var], label=var)

            self._draw_metrics_box(ax_sinal, metrics)

            ax_sinal.set_xlabel("Tempo (s)")
            ax_sinal.set_ylabel(self.y_label)
            ax_sinal.set_title(label)
            ax_sinal.legend(loc="best")
            ax_sinal.grid(True)

            if ax_erro is not None:
                self._plot_error_band(ax_erro, df, metrics)

            plt.tight_layout()

            safe_label = re.sub(r"[^A-Za-z0-9._-]+", "_", label).strip("._")
            output_path = output_dir / f"{safe_label}_response.png"
            plt.savefig(output_path)
            logger.info("Saved %s", output_path)
            plt.close(fig)

            output_paths.append(output_path)

        return output_paths


def _parse_args() -> "argparse.Namespace":
    parser = argparse.ArgumentParser(
        description="Plota variáveis de resposta no tempo (ScopeWizard/TwinCAT) "
        "para cada arquivo .csv de uma pasta, um PNG por arquivo."
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Pasta com os arquivos .csv de entrada.")
    parser.add_argument(
        "--var-filter", type=str, action="append", default=None,
        help="Substring para filtrar quais variáveis plotar (ex: 'Velo'). "
        "Pode ser repetido para múltiplos filtros: --var-filter Velo --var-filter Pos. "
        "Padrão: 'Velo'.",
    )
    parser.add_argument("--y-label", type=str, default="Velocidade (mm/s)", help="Rótulo do eixo Y.")
    parser.add_argument(
        "--setpoint", type=str, action="append", default=None,
        help="Setpoint para cálculo de overshoot/erro de regime, no formato "
        "'substring=valor' (ex: --setpoint Velo=10.0). Pode ser repetido para "
        "várias variáveis. Sem isso, as métricas não são calculadas.",
    )
    parser.add_argument(
        "--regime-delay", type=float, default=0.3,
        help="Tempo (segundos) após o instante do pico (overshoot) aguardado antes de "
        "começar a medir o regime permanente (padrão 0.3s).",
    )
    parser.add_argument(
        "--regime-window", type=float, default=0.2,
        help="Duração (segundos) da janela usada para calcular o valor de regime "
        "permanente, a partir do fim do atraso pós-pico (padrão 0.2s).",
    )
    parser.add_argument("--output-dir", type=str, default="./scope_view_pngs", help="Pasta de saída dos PNGs.")
    parser.add_argument("--show", action="store_true", help="Exibe os gráficos na tela além de salvar.")
    parser.add_argument(
        "--debug", action="store_true",
        help="Ativa logs detalhados de depuração do cálculo de regime permanente/overshoot.",
    )
    return parser.parse_args()


def _parse_setpoints(raw: Optional[List[str]]) -> Dict[str, float]:
    if not raw:
        return {}
    setpoints = {}
    for item in raw:
        chave, _, valor = item.partition("=")
        if not valor:
            raise ValueError(f"Setpoint inválido: '{item}'. Use o formato 'substring=valor'.")
        setpoints[chave] = float(valor)
    return setpoints


def main() -> None:
    args = _parse_args()

    if args.debug:
        logger.setLevel(logging.DEBUG)

    viz = ScopeVisualizer(
        data_dir=args.data_dir,
        var_filters=args.var_filter,
        y_label=args.y_label,
        setpoints=_parse_setpoints(args.setpoint),
        regime_delay_s=args.regime_delay,
        regime_window_s=args.regime_window,
    )
    viz.run(output_dir=args.output_dir)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()