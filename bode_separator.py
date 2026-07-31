"""
Separação de experimentos de resposta em frequência (Bode Plot) exportados
pelo TwinCAT em um único arquivo .csv, em arquivos individuais por bloco.

Autor original: Edélio Gabriel Magalhães de Jesus
Refatorado para estrutura orientada a objetos.

Uso:

python bodes_separation.py --input "./files/Bode Project_VEL_PAPU_TUNNING.csv" --output "./bode_files_PAPU"       
"""

import argparse
import csv
import re
import logging
from pathlib import Path
from typing import List, Tuple, Optional, Union

import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


class BodeSeparator:
    """
    Lê um arquivo .csv exportado do TwinCAT contendo múltiplos blocos de
    medição de resposta em frequência (Bode Plot) e os separa em arquivos
    .csv individuais, um por bloco/experimento.

    Uso:
        separator = BodeSeparator(
            input_file="./files/Bode Project_VEL_PAPU_TUNNING.csv",
            output_dir="./bode_files_PAPU",
        )
        separator.run()
    """

    def __init__(self, input_file: Union[str, Path], output_dir: Union[str, Path]):
        self.input_file = Path(input_file)
        self.output_dir = Path(output_dir)
        self._blocks: List[Tuple[str, pd.DataFrame]] = []

    # ---------- API pública ----------

    def run(self) -> None:
        """Executa o pipeline completo: carregar, parsear e salvar os blocos."""
        self._blocks = self._parse_blocks()
        self._save_blocks()

    @property
    def blocks(self) -> List[Tuple[str, pd.DataFrame]]:
        """Lista de tuplas (nome_do_bloco, DataFrame) resultantes do parsing."""
        return self._blocks

    # ---------- Métodos internos ----------

    @staticmethod
    def _make_safe_name(name: Optional[str], index: int) -> str:
        if not name:
            return f"block_{index + 1}"
        safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
        return safe_name or f"block_{index + 1}"

    def _parse_blocks(self) -> List[Tuple[str, pd.DataFrame]]:
        lines = self.input_file.read_text(encoding="utf-8-sig").splitlines()

        raw_blocks: List[Tuple[str, List[str], List[List[str]]]] = []
        current_name: Optional[str] = None
        current_header: Optional[List[str]] = None
        current_rows: List[List[str]] = []

        for line in lines:
            line_str = line.strip()
            if not line_str:
                continue

            # Início de um novo bloco (Name) ou metadados de registro (Record-)
            if line_str.startswith("Name") or line_str.startswith("Record-"):
                if current_header is not None and current_rows:
                    safe_name = self._make_safe_name(current_name, len(raw_blocks))
                    raw_blocks.append((safe_name, current_header, current_rows))
                    current_rows = []
                    current_header = None

                if line_str.startswith("Name"):
                    current_name = line_str.split(" ", 1)[1].strip()
                continue

            # Cabeçalho das colunas do bloco atual
            if line_str.startswith("Frequency"):
                current_header = next(csv.reader([line_str]))
                continue

            # Linhas de dados numéricos (frequência, ganho, fase)
            if current_header is not None and (line_str[0].isdigit() or line_str.startswith("-")):
                current_rows.append(next(csv.reader([line_str])))

        # Salva o último bloco do arquivo após sair do laço
        if current_header is not None and current_rows:
            safe_name = self._make_safe_name(current_name, len(raw_blocks))
            raw_blocks.append((safe_name, current_header, current_rows))

        if not raw_blocks:
            raise ValueError("Nenhum bloco de dados do Bode foi encontrado no arquivo.")

        return [
            (name, pd.DataFrame(rows, columns=header))
            for name, header, rows in raw_blocks
        ]

    def _save_blocks(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        for name, df in self._blocks:
            output_path = self.output_dir / f"{name}.csv"
            # Separador de tabulação para manter compatibilidade com scripts anteriores
            df.to_csv(output_path, sep="\t", index=False)
            logger.info("Saved %s", output_path)


def _parse_args() -> "argparse.Namespace":
    parser = argparse.ArgumentParser(
        description="Separa um arquivo Bode do TwinCAT (ou uma pasta inteira "
        "de arquivos) em CSVs individuais por bloco/experimento."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Caminho de um arquivo .csv, OU de uma pasta contendo vários "
        "arquivos .csv (use --batch nesse caso).",
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Pasta de saída. No modo --batch, cada arquivo de entrada gera "
        "uma subpasta dentro deste diretório.",
    )
    parser.add_argument(
        "--batch",
        action="store_true",
        help="Trata --input como uma pasta e processa todos os .csv dentro dela.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    input_path = Path(args.input)
    output_path = Path(args.output)

    if args.batch:
        csv_files = sorted(input_path.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(f"Nenhum .csv encontrado em {input_path}")

        for csv_file in csv_files:
            block_output_dir = output_path / csv_file.stem
            logger.info("Processando %s ...", csv_file.name)
            BodeSeparator(input_file=csv_file, output_dir=block_output_dir).run()
    else:
        BodeSeparator(input_file=input_path, output_dir=output_path).run()


if __name__ == "__main__":
    main()