"""
Esse script lê todos os arquivos .csv exportados pelo ScopeWizard (TwinCAT)
de uma pasta, contendo respostas no tempo (ex: resposta a um degrau), com
múltiplas variáveis dispostas lado a lado (cada uma com seu par de colunas
índice/tempo e valor), separadas por tab, aceitando vírgula ou ponto como
separador decimal.

Autor: Edélio Gabriel Magalhães de Jesus
Desenvolvido com auxílio de Inteligência Artificial
"""
import os
import glob
import io
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

# ---- Configuração ----
PASTA_DADOS = './scope_view_files'  # pasta com os .csv de resposta no tempo
OUTPUT_DIR = Path('./scope_view_pngs')
OUTPUT_DIR.mkdir(exist_ok=True)

# varre TODOS os .csv da pasta (sem filtro de nome)
arquivos_csv = sorted(glob.glob(os.path.join(PASTA_DADOS, '*.csv')))


def _eh_numero(campo):
    """True se o campo (string) representa um número, com decimal ',' ou '.'."""
    s = campo.strip().lstrip('-').replace(',', '.', 1)
    return s.replace('.', '', 1).isdigit()


def carregar_dados_step(arquivo):
    """
    Faz o parsing de um arquivo do ScopeWizard, com layout:

        (linhas de metadado do arquivo)
        Name    <var1>    Name    <var2>    Name    <var3>   ...
        (linhas de metadado da variável, opcionais)
        0   valor1_0   0   valor2_0   0   valor3_0
        2   valor1_1   2   valor2_1   2   valor3_1
        ...

    Retorna um DataFrame com colunas: Time_ms, <var1>, <var2>, <var3>, ...
    """
    with open(arquivo, 'r', encoding='utf-8-sig') as f:
        linhas = [l.rstrip('\n') for l in f]
    campos_por_linha = [l.split('\t') for l in linhas]

    # cabeçalho de variáveis: única linha que repete "Name" nas colunas pares
    idx_header = next(
        i for i, c in enumerate(campos_por_linha)
        if len(c) >= 4 and c[0] == 'Name' and c[2] == 'Name'
    )
    nomes_variaveis = [c.strip() for c in campos_por_linha[idx_header][1::2] if c.strip()]

    # início dos dados: primeira linha após o cabeçalho com valor numérico na 1ª coluna
    idx_dados = next(
        i for i in range(idx_header + 1, len(campos_por_linha))
        if _eh_numero(campos_por_linha[i][0])
    )

    texto_dados = '\n'.join(l for l in linhas[idx_dados:] if l.strip())
    decimal_sep = ',' if ',' in texto_dados.split('\n', 1)[0] else '.'

    df_raw = pd.read_csv(
        io.StringIO(texto_dados), sep='\t', header=None,
        decimal=decimal_sep, engine='python',
    )

    # coluna 0 = tempo (ms), comum a todas as variáveis (mesmo SampleTime)
    df = pd.DataFrame({'Time_ms': df_raw[0].astype(float)})
    for j, nome in enumerate(nomes_variaveis):
        df[nome] = df_raw[2 * j + 1].astype(float)

    return df


# ---- Carregamento ----
dfs, labels = [], []
for arquivo in arquivos_csv:
    nome_experimento = os.path.splitext(os.path.basename(arquivo))[0]
    try:
        df = carregar_dados_step(arquivo)
        dfs.append(df)
        labels.append(nome_experimento)
        print(f'{nome_experimento}: {len(df)} amostras carregadas ({list(df.columns[1:])})')
    except Exception as e:
        print(f'Erro ao processar o arquivo {nome_experimento}: {e}')

# ---- Plot: apenas variáveis de velocidade, sobrepostas ----
if dfs:
    for df, label in zip(dfs, labels):
        variaveis_velo = [c for c in df.columns if 'Velo' in c]
        if not variaveis_velo:
            print(f'{label}: nenhuma variável de velocidade encontrada, pulando.')
            continue

        tempo_s = df['Time_ms'] / 1000.0
        fig, ax = plt.subplots(figsize=(10, 5))
        for var in variaveis_velo:
            ax.plot(tempo_s, df[var], label=var)

        ax.set_xlabel('Tempo (s)')
        ax.set_ylabel('Velocidade (mm/s)')
        ax.set_title(label)
        ax.legend(loc='best')
        ax.grid(True)
        plt.tight_layout()

        output_path = OUTPUT_DIR / f'{label}_velo_response.png'
        plt.savefig(output_path)
        print(f'Saved {output_path}')
        plt.show()
else:
    print('Nenhum arquivo foi carregado com sucesso.')