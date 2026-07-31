"""
Plotagem comparativa do diagrama de Bode de várias curvas (arquivos .csv)
dentro de uma pasta, com cálculo de margens de ganho/fase e ordenação
automática pelo valor de um parâmetro extraído do nome do arquivo.

Autor original: Edélio Gabriel Magalhães de Jesus
Refatorado para estrutura orientada a objetos com auxílio de IA.

Uso:

python bode_visualizer.py --data-dir "./bode_files_PAPU/ --glob "*Vel*.csv" --system-part "Open-Loop" --output-dir "./bodes_pngs" --title "Comparative Bode Plot - Before tunning"
"""

import argparse
import io
import logging
import re
from pathlib import Path
from typing import List, Optional, Tuple, Union

import control as ct
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

class BodeVisualizer:
    """
    Carrega múltiplos arquivos .csv de resposta em frequência (Bode) de uma
    pasta, monta os sistemas em frequência (FRD), calcula margens de ganho e
    fase, e plota um diagrama de Bode comparativo, ordenado pelo valor de um
    parâmetro extraído do nome de cada arquivo (ex: Kp, Tn).

    Uso:
        viz = BodeVisualizer(
            data_dir="./bode_files_PAPU",
            file_glob="*Vel*.csv",
            system_part="Open-Loop",
            sort_param="kp",
        )
        viz.run(output_dir="./bodes_pngs", title="Comparative Bode Plot")
    """

    def __init__(
        self,
        data_dir: Union[str, Path],
        file_glob: str = "*.csv",
        system_part: str = "Open-Loop",
        sort_param: Optional[str] = None,
    ):
        self.data_dir = Path(data_dir)
        self.file_glob = file_glob
        self.system_part = system_part
        self.sort_param = sort_param

        self._systems: List[ct.FrequencyResponseData] = []
        self._labels: List[str] = []
        self._margins: List[Tuple[str, float, float, float, float]] = []

    # ---------- API pública ----------

    def run(self, output_dir: Union[str, Path], title: str = "Comparative Bode Plot") -> Path:
        """
        Executa o pipeline completo: carregar arquivos, montar sistemas,
        calcular margens, ordenar, plotar e salvar a figura.

        Retorna o caminho do PNG salvo.
        """
        self._load_all()

        if not self._systems:
            raise ValueError(
                f"Nenhum sistema foi carregado com sucesso a partir de "
                f"'{self.data_dir}' com o padrão '{self.file_glob}'."
            )

        if self.sort_param:
            self._sort_by_param(self.sort_param)

        return self._plot_and_save(output_dir=Path(output_dir), title=title)

    @property
    def systems(self) -> List[ct.FrequencyResponseData]:
        """Lista de sistemas FRD carregados (após `run`)."""
        return self._systems

    @property
    def labels(self) -> List[str]:
        """Nomes dos experimentos correspondentes a `systems`."""
        return self._labels

    @property
    def margins(self) -> List[Tuple[str, float, float, float, float]]:
        """Lista de (nome, margem_ganho, margem_fase, freq_cruz_ganho, freq_cruz_fase)."""
        return self._margins

    # ---------- Métodos internos ----------

    @staticmethod
    def _load_csv(path: Path) -> pd.DataFrame:
        with open(path, "r") as f:
            linhas = [linha.strip().strip('"') for linha in f]

        idx_header = next(i for i, l in enumerate(linhas) if l.startswith("Frequency"))
        texto_dados = "\n".join(linhas[idx_header:])

        return pd.read_csv(io.StringIO(texto_dados), sep="\t")

    def _load_all(self) -> None:
        self._systems = []
        self._labels = []
        self._margins = []

        arquivos_csv = sorted(self.data_dir.glob(self.file_glob))

        for arquivo in arquivos_csv:
            nome_experimento = arquivo.stem
            nome_sistema = nome_experimento.replace(".", "_")

            try:
                df = self._load_csv(arquivo)

                gain = df[f"{self.system_part}-Gain"].to_numpy()
                phase = df[f"{self.system_part}-Phase"].to_numpy()
                frequency = df["Frequency"].to_numpy()

                mag_linear = 10 ** (gain / 20)
                phase_rad = np.deg2rad(phase)
                omega = 2 * np.pi * frequency
                response_complex = mag_linear * np.exp(1j * phase_rad)

                sys_frd = ct.frd(response_complex, omega, name=nome_sistema)
                gm, pm, wcg, wcp = ct.margin(sys_frd)

                logger.info(
                    "Arquivo: %s | Margem de Ganho = %s | Margem de Fase = %s",
                    nome_experimento, gm, pm,
                )

                self._systems.append(sys_frd)
                self._labels.append(nome_experimento)
                self._margins.append((nome_experimento, gm, pm, wcg, wcp))

            except Exception as e:
                logger.warning("Erro ao processar o arquivo %s: %s", nome_experimento, e)

    def _extract_param_value(self, label: str, param: str) -> float:
        """Extrai o valor numérico de um parâmetro do nome do arquivo.

        Assume o padrão '_{param}_<numero>' (ex: '_kp_120', '_tn_0.5').
        Retorna 0.0 se o parâmetro não for encontrado no nome.
        """
        busca = re.search(rf"_{re.escape(param)}_([0-9.]+)", label, flags=re.IGNORECASE)
        return float(busca.group(1)) if busca else 0.0

    def _sort_by_param(self, param: str) -> None:
        pares_ordenados = sorted(
            zip(self._systems, self._labels),
            key=lambda par: self._extract_param_value(par[1], param),
        )
        self._systems, self._labels = (list(t) for t in zip(*pares_ordenados))

    def _plot_and_save(self, output_dir: Path, title: str) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)

        ct.bode_plot(
            self._systems, dB=True, deg=True, label=self._labels, display_margins=True
        )

        handles, all_labels = plt.gcf().axes[0].get_legend_handles_labels()

        by_label = {}
        for h, l in zip(handles, all_labels):
            if l in self._labels and l not in by_label:
                by_label[l] = h

        if plt.gcf().axes[0].legend_ is not None:
            plt.gcf().axes[0].legend_.remove()

        plt.gcf().legend(
            by_label.values(), by_label.keys(), loc="center left", bbox_to_anchor=(1.02, 0.5)
        )
        plt.subplots_adjust(right=0.75)
        plt.suptitle(title, y=0.98)

        safe_glob = re.sub(r"[^A-Za-z0-9._-]+", "_", self.file_glob).strip("._")
        output_path = output_dir / f"{title}_Bode_Plots.png"
        plt.savefig(output_path, bbox_inches="tight")
        logger.info("Saved %s", output_path)

        return output_path


def _parse_args() -> "argparse.Namespace":
    parser = argparse.ArgumentParser(
        description="Plota um diagrama de Bode comparativo a partir de vários "
        "arquivos .csv de uma pasta."
    )
    parser.add_argument("--data-dir", type=str, required=True, help="Pasta com os arquivos .csv de entrada.")
    parser.add_argument(
        "--glob", type=str, default="*.csv",
        help="Padrão glob para filtrar os arquivos dentro da pasta (ex: '*Vel*.csv').",
    )
    parser.add_argument(
        "--system-part", type=str, default="Open-Loop",
        help="Prefixo das colunas de ganho/fase a usar (ex: 'Open-Loop', 'Process').",
    )
    parser.add_argument(
        "--sort-param", type=str, default=None,
        help="Nome do parâmetro a extrair do nome do arquivo para ordenação (ex: 'kp', 'tn'). "
        "Assume o padrão '_{param}_<numero>' no nome do arquivo.",
    )
    parser.add_argument("--output-dir", type=str, default="./bodes_pngs", help="Pasta de saída do PNG.")
    parser.add_argument("--title", type=str, default="Comparative Bode Plot", help="Título do gráfico.")
    parser.add_argument("--show", action="store_true", help="Exibe o gráfico na tela além de salvar.")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()

    viz = BodeVisualizer(
        data_dir=args.data_dir,
        file_glob=args.glob,
        system_part=args.system_part,
        sort_param=args.sort_param,
    )
    viz.run(output_dir=args.output_dir, title=args.title)

    if args.show:
        plt.show()


if __name__ == "__main__":
    main()