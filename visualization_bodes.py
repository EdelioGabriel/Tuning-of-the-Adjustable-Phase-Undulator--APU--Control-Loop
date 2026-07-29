"""
Esse script tem como objetivo plotar o diagrama de Bode das curvas de todos os arquivos de uma pasta

Autor: Edélio Gabriel Magalhães de Jesus

Desenvolvido com auxílio de Inteligência Aritificial
"""
import os
import glob
import io
import pandas as pd
import control as ct
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

TIPO = 'Pos'
PASTA_DADOS = './bode_files_PAPU'
OUTPUT_DIR = Path('./bodes_pngs')

arquivos_csv = glob.glob(os.path.join(PASTA_DADOS, f'*{TIPO}*.csv'))

def carregar_dados(arquivo):
    with open(arquivo, 'r') as f:
        linhas = [linha.strip().strip('"') for linha in f]

    idx_header = next(i for i, l in enumerate(linhas) if l.startswith('Frequency'))
    texto_dados = '\n'.join(linhas[idx_header:])

    df = pd.read_csv(io.StringIO(texto_dados), sep='\t')
    return df

sistemas = []
labels = []

for arquivo in arquivos_csv:
    nome_experimento = os.path.splitext(os.path.basename(arquivo))[0]  # sem extensão
    nome_sistema = nome_experimento.replace('.', '_')  # sem pontos

    try:
        df = carregar_dados(arquivo)

        gain = df['Process-Gain'].to_numpy()
        phase = df['Process-Phase'].to_numpy()
        frequency = df['Frequency'].to_numpy()

        mag_linear = 10 ** (gain / 20)
        phase_rad = np.deg2rad(phase)
        omega = 2 * np.pi * frequency
        response_complex = mag_linear * np.exp(1j * phase_rad)

        sys_frd = ct.frd(response_complex, omega, name=nome_sistema)

        gm, pm, wcg, wcp = ct.margin(sys_frd)

        print(f'Margem de Ganho = {gm}\n Marrgem de Fase = {pm}\n -----------------')
        sistemas.append(sys_frd)
        labels.append(nome_experimento)  # o label do gráfico pode manter o nome original

    except Exception as e:
        print(f'Erro ao processar o arquivo {nome_experimento}: {e}')

if sistemas:
    ct.bode_plot(sistemas, dB=True, deg=True, label=labels)
    plt.gcf().axes[0].legend(loc='lower left')
    plt.tight_layout()

    output_path = OUTPUT_DIR / f'{TIPO}_Bode_Plots.png'
    plt.savefig(output_path)
    print(f"Saved {output_path}")

    plt.show()
else:
    print('Nenhum sistema foi carregado com sucesso.')
