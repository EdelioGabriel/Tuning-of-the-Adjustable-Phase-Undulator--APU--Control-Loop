"""
Agrega os erros RMS de identificação de função de transferência a partir dos
arquivos .json exportados por tf_identifier.py, optuna_tf_identifier.py e
vector_fitting_identifier.py, agrupa por método (identificado pelo prefixo do
nome do arquivo) e plota um único gráfico comparativo com boxplots.

Convenção de prefixo -> método (definida pelos próprios scripts de ajuste):
  LS_...json      -> Mínimos Quadrados (tf_identifier.py)
  OPTUNA_...json  -> Optuna + AICc (optuna_tf_identifier.py)
  VF_...json      -> Vector Fitting (vector_fitting_identifier.py)

Cada JSON pode guardar o erro final sob uma chave diferente:
  "rms_error"                  -> LS / OPTUNA
  "rms_error_vector_fitting"   -> VF
O script tenta ambas, nessa ordem.

Uso:
  python analise_erros_metodos.py --input-dir ./tfs_json_PAPU --output ./comparativo_rms.png
"""

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Union

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


# =============================================================================
# Identificação do método a partir do nome do arquivo
# =============================================================================

# Ordem importa: prefixos mais específicos primeiro, caso um dia surja
# ambiguidade (ex.: "OPTUNA_LS_..." não existe hoje, mas por segurança).
PREFIXO_PARA_METODO: Dict[str, str] = {
    "OPTUNA": "Optuna + AICc",
    "VF": "Vector Fitting",
    "LS": "Mínimos Quadrados",
}

CHAVES_RMS = ("rms_error", "rms_error_vector_fitting")


def identificar_metodo(nome_arquivo: str) -> str:
    """Identifica o método a partir do prefixo (antes do primeiro '_') do
    nome do arquivo. Se o prefixo não for reconhecido, o próprio prefixo é
    usado como rótulo (para não descartar dados silenciosamente)."""
    return PREFIXO_PARA_METODO.get(extrair_prefixo(nome_arquivo), extrair_prefixo(nome_arquivo))


def extrair_prefixo(nome_arquivo: str) -> str:
    """Prefixo (antes do primeiro '_'), em maiúsculas, usado tanto para
    identificar o método quanto para filtrar por --prefixes."""
    return nome_arquivo.split("_")[0].upper()


def extrair_rms(dados: dict, caminho: Path) -> Optional[float]:
    """Extrai o erro RMS de um dict de JSON já carregado, tentando as chaves
    conhecidas nessa ordem. Retorna None (e loga um aviso) se nenhuma bater."""
    for chave in CHAVES_RMS:
        if chave in dados:
            return float(dados[chave])
    logger.warning("Aviso: %s não contém nenhuma das chaves de RMS esperadas %s -- ignorado.", caminho.name, CHAVES_RMS)
    return None


# =============================================================================
# Coleta e agregação
# =============================================================================

class ColetorErrosRMS:
    """
    Varre uma pasta de JSONs de TF ajustada, agrupa os erros RMS por método
    (via prefixo do nome do arquivo) e calcula estatísticas por grupo.

    Se `prefixos_desejados` for informado, apenas arquivos cujo prefixo
    (case-insensitive) esteja nessa lista são processados -- os demais são
    ignorados silenciosamente (contabilizados e reportados no final). Se
    None, todos os arquivos .json da pasta são processados.
    """

    def __init__(self, input_dir: Union[str, Path], prefixos_desejados: Optional[List[str]] = None):
        self.input_dir = Path(input_dir)
        self.prefixos_desejados = (
            {p.upper() for p in prefixos_desejados} if prefixos_desejados else None
        )
        self.erros_por_metodo: Dict[str, List[float]] = defaultdict(list)
        self.n_ignorados_prefixo = 0

    def coletar(self) -> "ColetorErrosRMS":
        arquivos_json = sorted(self.input_dir.glob("*.json"))
        if not arquivos_json:
            raise FileNotFoundError(f"Nenhum arquivo .json encontrado em {self.input_dir}")

        for caminho in arquivos_json:
            prefixo = extrair_prefixo(caminho.name)

            if self.prefixos_desejados is not None and prefixo not in self.prefixos_desejados:
                self.n_ignorados_prefixo += 1
                continue

            with open(caminho, "r") as f:
                dados = json.load(f)

            rms = extrair_rms(dados, caminho)
            if rms is None:
                continue

            metodo = PREFIXO_PARA_METODO.get(prefixo, prefixo)
            self.erros_por_metodo[metodo].append(rms)

        if self.n_ignorados_prefixo:
            logger.info(
                "%d arquivo(s) ignorado(s) por não estarem na lista de prefixos: %s",
                self.n_ignorados_prefixo, sorted(self.prefixos_desejados),
            )

        if not self.erros_por_metodo:
            raise RuntimeError("Nenhum erro RMS pôde ser extraído dos JSONs encontrados (verifique os prefixos informados).")

        return self

    def estatisticas(self) -> Dict[str, Dict[str, float]]:
        """Retorna {metodo: {"media": ..., "desvio_padrao": ..., "n": ...}}."""
        stats = {}
        for metodo, erros in self.erros_por_metodo.items():
            arr = np.array(erros)
            stats[metodo] = {
                "media": float(np.mean(arr)),
                "desvio_padrao": float(np.std(arr, ddof=1)) if len(arr) > 1 else 0.0,
                "n": len(arr),
            }
        return stats

    def logar_estatisticas(self) -> None:
        logger.info("\n=== ERRO RMS POR MÉTODO ===")
        for metodo, s in sorted(self.estatisticas().items(), key=lambda kv: kv[1]["media"]):
            logger.info(
                "%-20s | n=%3d | média=%.6e | desvio padrão=%.6e",
                metodo, s["n"], s["media"], s["desvio_padrao"],
            )


# =============================================================================
# Plot
# =============================================================================

# Paleta fixa por método -- assim a cor de cada método fica consistente
# entre execuções diferentes, mesmo que o conjunto de métodos presentes mude.
PALETA_METODOS: Dict[str, str] = {
    "Mínimos Quadrados": "#4C72B0",
    "Optuna + AICc": "#DD8452",
    "Vector Fitting": "#55A868",
}
COR_FALLBACK = "#8172B2"


def plotar_boxplots(erros_por_metodo: Dict[str, List[float]], output_path: Path, log_scale: bool = True) -> None:
    """Gera um único gráfico com um boxplot por método, ordenados pela
    mediana (do melhor para o pior)."""
    metodos_ordenados = sorted(
        erros_por_metodo.keys(),
        key=lambda m: np.median(erros_por_metodo[m]),
    )
    dados_plot = [erros_por_metodo[m] for m in metodos_ordenados]
    cores = [PALETA_METODOS.get(m, COR_FALLBACK) for m in metodos_ordenados]

    plt.rcParams.update({
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
    })

    fig, ax = plt.subplots(figsize=(9, 6))

    bp = ax.boxplot(
        dados_plot,
        tick_labels=metodos_ordenados,
        showmeans=True,
        patch_artist=True,
        widths=0.55,
        meanprops=dict(marker="D", markerfacecolor="white", markeredgecolor="black",
                       markersize=6, zorder=4),
        medianprops=dict(color="black", linewidth=1.6),
        whiskerprops=dict(color="#444444", linewidth=1.1),
        capprops=dict(color="#444444", linewidth=1.1),
        flierprops=dict(marker="o", markerfacecolor="#444444", markeredgecolor="none",
                         markersize=4, alpha=0.5),
        zorder=3,
    )
    for patch, cor in zip(bp["boxes"], cores):
        patch.set_facecolor(cor)
        patch.set_alpha(0.75)
        patch.set_edgecolor("#333333")
        patch.set_linewidth(1.1)

    if log_scale:
        ax.set_yscale("log")

    # Anotações de n e média acima de cada caixa
    y_min, y_max = ax.get_ylim()
    offset = (np.log10(y_max) - np.log10(y_min)) * 0.04 if log_scale else (y_max - y_min) * 0.04
    for i, metodo in enumerate(metodos_ordenados, start=1):
        erros = erros_por_metodo[metodo]
        topo = max(erros)
        y_anot = topo * (10 ** offset) if log_scale else topo + offset
        ax.text(
            i, y_anot, f"n={len(erros)}\nμ={np.mean(erros):.3e}",
            ha="center", va="bottom", fontsize=8.5, color="#333333",
        )

    ax.set_ylim(top=(y_max * (10 ** (offset * 3.5)) if log_scale else y_max + offset * 3.5))
    ax.set_ylabel("Erro RMS" + (" (escala log)" if log_scale else ""))
    ax.set_title("Comparação do Erro RMS por Método de Identificação")
    ax.grid(True, which="major" if log_scale else "both", axis="y", linestyle="--", alpha=0.4, zorder=0)
    ax.set_axisbelow(True)

    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(axis="x", labelsize=10)

    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    logger.info("\nGráfico salvo em %s", output_path)
    plt.close(fig)


# =============================================================================
# CLI
# =============================================================================

def _parse_args() -> "argparse.Namespace":
    parser = argparse.ArgumentParser(
        description="Agrega e compara erros RMS de identificação de TF por método, a partir dos JSONs exportados."
    )
    parser.add_argument("--input-dir", type=str, required=True, help="Pasta contendo os arquivos .json.")
    parser.add_argument(
        "--prefixes", type=str, nargs="+", default=None,
        help="Lista de prefixos a varrer (ex.: --prefixes LS OPTUNA VF). "
        "Se omitido, processa todos os .json da pasta.",
    )
    parser.add_argument("--output", type=str, default="./comparativo_rms_boxplot.png", help="Caminho do PNG de saída.")
    parser.add_argument("--linear-scale", action="store_true", help="Usa escala linear no eixo Y (padrão: log).")
    parser.add_argument("--show", action="store_true", help="Exibe o gráfico na tela além de salvar.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    coletor = ColetorErrosRMS(args.input_dir, prefixos_desejados=args.prefixes).coletar()
    coletor.logar_estatisticas()
    plotar_boxplots(coletor.erros_por_metodo, Path(args.output), log_scale=not args.linear_scale)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()