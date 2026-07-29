import json
import control as ct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

NOME_ARQUIVO = "./tfs_json_PAPU/Open-Loop_TF_Id_1_Vel_NC_kp_586_Tn_15__1.csv.json"

BODE_NAME_FILE = "./bode_files_PAPU/Id_1_Vel_NC_kp_586_Tn_15__1.csv"

try:
    with open(NOME_ARQUIVO, 'r') as arquivo:
        dados = json.load(arquivo)

        sys_tf = ct.tf(dados['num'], dados['den'])
        print(f"Modelo lido do arquivo '{NOME_ARQUIVO}':")
        print(sys_tf)

except FileNotFoundError:
    raise FileNotFoundError(f"Arquivo '{NOME_ARQUIVO}' não encontrado.")

# 2. Definição do vetor de tempo em SEGUNDOS, mas com resolução para milissegundos
# 0 a 25 milissegundos (0.025s) com 5000 pontos para garantir precisão numérica
t = np.linspace(0, 0.040, 5000)

# 3. Construção do sinal com os degraus (Janela de 8 ms ativa)
U = np.zeros_like(t)

# Transforma os limites para segundos para aplicar no vetor t:
# t >= 2 ms e t < 10 ms (Janela exata de 8 ms de degrau positivo)
U[(t >= 0.002) & (t < 0.015)] = 1.0

# t >= 10 ms e t < 18 ms (Janela de 8 ms de degrau negativo para compensar o integrador)
U[(t >= 0.015) & (t < 0.018)] = -1.0

# t >= 18 ms o sinal retorna a 0

# 4. Executa a simulação da resposta forçada
response = ct.forced_response(sys_tf, T=t, U=U)

# 5. Plotagem dos resultados convertendo o eixo X para ms (visualização mais fácil)
t_ms = t * 1000  # Converte segundos para milissegundos apenas para o gráfico

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Gráfico da Entrada (U)
ax1.plot(t_ms, U, color='orange', linewidth=2, label='Entrada (Pulsos de 8 ms)')
ax1.set_ylabel('Entrada (U)')
ax1.grid(True, linestyle="--", linewidth=0.5)
ax1.legend()
ax1.set_title('Simulação de Alta Velocidade (Escala de Milissegundos)')

# Gráfico da Saída (Y)
ax2.plot(t_ms, response.outputs, color='blue', linewidth=2, label='Saída do Sistema')
ax2.set_xlabel('Tempo (milissegundos)')
ax2.set_ylabel('Saída (Y)')
ax2.grid(True, linestyle="--", linewidth=0.5)
ax2.legend()

plt.tight_layout()
plt.show()