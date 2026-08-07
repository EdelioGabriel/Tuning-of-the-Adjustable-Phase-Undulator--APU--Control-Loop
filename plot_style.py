"""
plot_style.py — estilo padronizado para todos os plots do projeto.
Chame apply_style() uma vez no início de cada script (após os imports).
"""

import matplotlib as mpl
import matplotlib.pyplot as plt

DPI = 300
FIGSIZE_BODE = (9, 6)      # figuras de 2 subplots (mag/phase)
FIGSIZE_SINGLE = (9, 5)    # figuras de 1 subplot
COLOR_CYCLE = ["#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B2", "#937860"]
BOX_ASPECT = 0.4

def apply_style() -> None:
    mpl.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": DPI,
        "savefig.bbox": "tight",
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.titleweight": "bold",
        "axes.labelsize": 11,
        "legend.fontsize": 11,
        "legend.frameon": True,
        "legend.framealpha": 0.9,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "lines.linewidth": 1.6,
        "axes.grid": False,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.prop_cycle": mpl.cycler(color=COLOR_CYCLE),
    })


def clean_grid(ax, axis: str = "both") -> None:
    """Grade leve: só linhas principais (major), baixa opacidade."""
    ax.grid(True, which="major", axis=axis, linestyle="--", linewidth=0.5, alpha=0.35)
    ax.grid(False, which="minor")


def set_box_aspect(ax, aspect: float = BOX_ASPECT) -> None:
    """Fixa a proporção do retângulo de dados do eixo, independente de
    legenda, título ou tamanho de labels."""
    ax.set_box_aspect(aspect)


def standardize_bode_axes(fig, aspect: float = BOX_ASPECT) -> None:
    """Grade leve + retângulo de dados padronizado em todos os eixos de uma
    figura de Bode. Substitui strip_bode_grid()."""
    for ax in fig.axes:
        clean_grid(ax)
        set_box_aspect(ax, aspect)


def savefig_hq(path, fig=None) -> None:
    (fig or plt).savefig(path, dpi=DPI, bbox_inches="tight")