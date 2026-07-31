import json
import control as ct
import numpy as np
import matplotlib.pyplot as plt

NOME_ARQUIVO = "./tfs_json_PAPU/VF_process_TF_Id_1_Vel_NC_kp_586_Tn_15__1.csv.json"

try:
    with open(NOME_ARQUIVO, 'r') as arquivo:
        dados = json.load(arquivo)

        sys_tf = ct.tf(dados['num'], dados['den'])
        print(f"Modelo lido do arquivo '{NOME_ARQUIVO}':")
        print(sys_tf)

except FileNotFoundError:
    raise FileNotFoundError(f"Arquivo '{NOME_ARQUIVO}' não encontrado.")

# 2. Definição do vetor de tempo em SEGUNDOS
# Janela maior (ex: 0 a 200 ms) para dar tempo do sistema se estabilizar
t = np.linspace(0, 2, 5000)

# 3. Construção do sinal com UM ÚNICO degrau (step único)
U = np.zeros_like(t)

# Degrau sobe em t = 2 ms e permanece em 1.0 até o final da simulação
U[t >= 0.5] = 1.0

sys_tf_sintonizada = sys_tf * 1
# 4. Executa a simulação da resposta forçada
response = ct.forced_response(sys_tf_sintonizada, T=t, U=U)

# 5. Plotagem dos resultados convertendo o eixo X para ms
t_ms = t * 1000

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)

# Gráfico da Entrada (U)
ax1.plot(t_ms, U, color='orange', linewidth=2, label='Entrada (Degrau único)')
ax1.set_ylabel('Entrada (U)')
ax1.grid(True, linestyle="--", linewidth=0.5)
ax1.legend()
ax1.set_title('Resposta ao Degrau - Análise de Estabilização')

# Gráfico da Saída (Y)
ax2.plot(t_ms, response.outputs, color='blue', linewidth=2, label='Saída do Sistema')
ax2.set_xlabel('Tempo (milissegundos)')
ax2.set_ylabel('Saída (Y)')
ax2.grid(True, linestyle="--", linewidth=0.5)
ax2.legend()

plt.tight_layout()
plt.show()