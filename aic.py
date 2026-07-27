import pandas as pd
import numpy as np
import control as ct
import glob
import os
import io
import json

NOME_ARQUIVO = './bode_files_PAPU/Id_1_Vel_NC_kp_586_Tn_15__1.csv'
PASTA_DADOS = './bode_files_PAPU'
arquivos_csv = glob.glob(os.path.join(PASTA_DADOS, '*.csv'))

def carregar_dados(arquivo):
    with open(arquivo, 'r') as f:
        linhas = [linha.strip().strip('"') for linha in f]

    idx_header = next(i for i, l in enumerate(linhas) if l.startswith('Frequency'))
    texto_dados = '\n'.join(linhas[idx_header:])

    df = pd.read_csv(io.StringIO(texto_dados), sep='\t')
    return df

# 1. Carregar os dados reais do diagrama de Bode
df = carregar_dados(NOME_ARQUIVO)

# Certifique-se de que os nomes das colunas batem com o seu CSV (ex: 'Frequency (Hz)', 'Magnitude (dB)', 'Phase (deg)')
freq_hz = df['Frequency'].values
mag_db = df['Open-Loop-Gain'].values
fase_deg = df['Open-Loop-Phase'].values

# Converter dados reais para números complexos (Ganho linear e Fase em radianos)
omega = 2 * np.pi * freq_hz                     # Frequência em rad/s
mag_linear = 10 ** (mag_db / 20)                # Converte dB para linear
fase_rad = np.radians(fase_deg)                 # Converte graus para radianos
resp_real_complexa = mag_linear * np.exp(1j * fase_rad)

# 2. Carregar a Função de Transferência do JSON
NOME_ARQUIVO_TF = 'TF_Id_1_Vel_NC_kp_586_Tn_15__1.csv.json'
with open(NOME_ARQUIVO_TF, 'r') as arquivo:
    dados = json.load(arquivo)

sys_tf = ct.tf(dados['num'], dados['den'])
print(f"Modelo lido do arquivo '{NOME_ARQUIVO_TF}':")
print(sys_tf)

# 3. Calcular a resposta em frequência teórica do modelo estimado
# ct.frequency_response retorna os ganhos complexos nas frequências desejadas
_, _, resp_modelo_complexa = ct.frequency_response(sys_tf, omega)
# Remove dimensões extras que a biblioteca control gera na matriz complexa
resp_modelo_complexa = resp_modelo_complexa.squeeze() 

# 4. Cálculo do Resíduo Quadrático (RSS) considerando Magnitude e Fase (Erro Complexo)
erros = resp_real_complexa - resp_modelo_complexa
rss = np.sum(np.abs(erros)**2)

# 5. Cálculo do AICc para Amostras Pequenas
N = len(freq_hz)  # No seu caso, 50

# Número de parâmetros (K): coeficientes do numerador + denominador do JSON
# Nota: removemos zeros à esquerda se houverem, avaliando o tamanho real dos vetores
K = len(dados['num']) + len(dados['den']) 

# Cálculo matemático do AIC adaptado para RSS e a correção para pequenos datasets (AICc)
aic = N * np.log(rss / N) + 2 * K
aicc = aic + (2 * K * (K + 1)) / (N - K - 1)

print("\n--- Avaliação Estatística (Amostras Pequenas) ---")
print(f"Número de pontos (N): {N}")
print(f"Número de parâmetros estimados (K): {K}")
print(f"Soma dos Resíduos Quadráticos (RSS): {rss:.6f}")
print(f"Valor do AICc: {aicc:.2f}")
