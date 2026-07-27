"""
Esse script tem como objetivo separar os experimentos contidos em um único arquivo .csv exportado diretamente do TwinCAT

Autor: Edélio Gabriel Magalhães de Jesus

Desenvolvido com auxílio de IA
"""

import csv
import re
from pathlib import Path
import pandas as pd

# =========================================================
# SEPARAÇÃO DOS BODES EM ARQUIVOS CSV INDIVIDUAIS
# =========================================================
FILE_NAME = "./files/BodeProject_PAPU_1.csv"
OUTPUT_DIR = Path("./bode_plots_PAPU")

def make_safe_name(name: str, index: int) -> str:
    if not name:
        return f"block_{index + 1}"
    safe_name = re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("._")
    return safe_name or f"block_{index + 1}"

def load_bode_table(path: str | Path = FILE_NAME):
    lines = Path(path).read_text(encoding="utf-8-sig").splitlines()
    blocks = []
    
    current_name = None
    current_header = None
    current_rows = []

    for line in lines:
        line_str = line.strip()
        if not line_str:
            continue
            
        # Detectou o início de um novo bloco (Name) ou metadados de registro (Record-)
        if line_str.startswith("Name") or line_str.startswith("Record-"):
            # Se já tínhamos dados acumulados de um bloco anterior, salva ele primeiro!
            if current_header is not None and current_rows:
                safe_name = make_safe_name(current_name, len(blocks))
                blocks.append((safe_name, current_header, current_rows))
                current_rows = []
                current_header = None  # Reseta o header para o novo bloco
            
            # Só depois de salvar o anterior, atualiza o nome do novo bloco
            if line_str.startswith("Name"):
                current_name = line_str.split(" ", 1)[1].strip()
            continue
            
        # Detectou o cabeçalho das colunas do bloco atual
        if line_str.startswith("Frequency"):
            current_header = next(csv.reader([line_str]))
            continue
            
        # Captura as linhas de dados numéricos (frequência, ganho, fase)
        if current_header is not None and (line_str[0].isdigit() or line_str.startswith("-")):
            current_rows.append(next(csv.reader([line_str])))

    # Não esquecer de salvar o ÚLTIMO bloco do arquivo após sair do laço
    if current_header is not None and current_rows:
        safe_name = make_safe_name(current_name, len(blocks))
        blocks.append((safe_name, current_header, current_rows))

    if not blocks:
        raise ValueError("Nenhum bloco de dados do Bode foi encontrado no arquivo.")

    return [(name, pd.DataFrame(rows, columns=header)) for name, header, rows in blocks]

# Garante que a pasta de destino exista
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

blocks = load_bode_table(FILE_NAME)
for name, df in blocks:
    output_path = OUTPUT_DIR / f"{name}.csv"
    # Salva usando separador de tabulação para manter compatibilidade com seu script anterior
    df.to_csv(output_path, sep='\t', index=False)
    print(f"Saved {output_path}")
