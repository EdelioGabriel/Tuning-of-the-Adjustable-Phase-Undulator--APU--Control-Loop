import json
import control as ct
import numpy as np
import matplotlib.pyplot as plt
import glob 
import os
from pathlib import Path

PARTE_DO_SISTEMA = 'open-loop' 
OTIMIZADOR = 'LS'
PASTA_DADOS = './tfs_json_PAPU'
VARIAVEL = 'Vel'
LIMITE_DESVIO_PCT = 0.10  # 10% de tolerância para o desvio padrão usar a média
OUTPUT_DIR = Path('./tunning_results')
OUTPUT_PATH = OUTPUT_DIR / f'TUNNING_{OTIMIZADOR}s_{PARTE_DO_SISTEMA.lower()}_TFs'

arquivos_json = glob.glob(os.path.join(PASTA_DADOS, f'{OTIMIZADOR}*{PARTE_DO_SISTEMA}*{VARIAVEL}*.json'))

def carregar_tf(caminho_arquivo):
    with open(caminho_arquivo, 'r') as f:
        dados = json.load(f)
        sys_tf = ct.tf(dados['num'], dados['den'])
        return ct.minreal(sys_tf, verbose=False)

# 1. Carrega todas as funções de transferência
sys_tfs = [carregar_tf(arq) for arq in arquivos_json]

# Vetor de frequências robusto para evitar o bug do 'inf' no cálculo numérico
omega_vetor = np.logspace(-3, 4, 1500)

margens_ganho_db = []
margens_fase_deg = []

print("--- RELATÓRIO DE MARGENS INDIVIDUAIS ---")
for i, tf in enumerate(sys_tfs):
    gm, pm, _, _ = ct.margin(tf)
    gm_db = 20 * np.log10(gm) if gm > 0 else np.nan
    
    margens_ganho_db.append(gm_db)
    margens_fase_deg.append(pm)
    print(f"Modelo {i+1}: MG = {gm_db:.2f} dB | MF = {pm:.2f}°")

# 2. Cálculos Estatísticos (Média e Desvio Padrão)
media_mg = np.nanmean(margens_ganho_db)
desvio_mg = np.nanstd(margens_ganho_db)

media_mf = np.nanmean(margens_fase_deg)
desvio_mf = np.nanstd(margens_fase_deg)

print("\n--- ANÁLISE ESTATÍSTICA ---")
print(f"Média das Margens  -> MG: {media_mg:.2f} dB | MF: {media_mf:.2f}°")
print(f"Desvio Padrão      -> MG: {desvio_mg:.2f} dB | MF: {desvio_mf:.2f}°")

# 3. Validação Estatística para o Tuning
# Verifica o desvio em relação à escala da média (Coeficiente de Variação)
desvio_aceitavel = (desvio_mg / media_mg < LIMITE_DESVIO_PCT) and (desvio_mf / media_mf < LIMITE_DESVIO_PCT)

print("\n--- PROCESSO DE TUNING ---")
if desvio_aceitavel:
    print("Validação Estatística Sucedida: Baixa variação entre os modelos.")
    print("Executando AJUSTE ÚNICO baseado na resposta média.")
    
    # Alvo de Margem de Ganho segura é 9.8 dB (cerca de 10 dB)
    # Calculamos quanto precisamos somar ou subtrair da média para atingir o alvo
    ALVO_MG = 10
    delta_db = media_mg - ALVO_MG
    K_adj_medio = 10**(delta_db / 20)
    
    print(f"   Fator K_adj calculado para a média: {K_adj_medio:.3f}")
    
    # Aplica o ganho médio na primeira FT da lista como referência de teste
    sys_final = sys_tfs[0] * K_adj_medio
    gm_f, pm_f, _, _ = ct.margin(sys_final)
    print(f"   Resultado esperado -> Nova MG: {20*np.log10(gm_f):.2f} dB | Nova MF: {pm_f:.2f}°")
    ct.bode_plot(sys_final, omega=omega_vetor, dB=True, Hz=False, label=f'Modelo médio', display_margins=True)
    plt.suptitle('Comparative Bode Plot')
    plt.savefig(f'{OUTPUT_PATH}_DOMINIO_DADOS.png')
    print(f"Saved {OUTPUT_PATH}")
    plt.show()

else:
    print("Validação Estatística Recusada: Modelos divergem muito entre si.")
    print("Executando 3 AJUSTES distintos para cenários diferentes (Pessimista, Médio, Otimista):")
    
    # Cenário 1: Baseado no Pior Caso (Menor Margem de Ganho encontrada da lista)
    pior_mg = np.nanmin(margens_ganho_db)
    K_adj_pessimista = 10**((pior_mg - 10) / 20)
    
    # Cenário 2: Baseado puramente na Média Matemática
    K_adj_medio = 10**((media_mg - 10) / 20)
    
    # Cenário 3: Baseado no Melhor Caso (Maior Margem de Ganho encontrada da lista)
    melhor_mg = np.nanmax(margens_ganho_db)
    K_adj_otimista = 10**((melhor_mg - 10) / 20)
    
    print(f"   1. K_adj Conservador (Pior Caso - MG {pior_mg:.1f} dB): {K_adj_pessimista:.3f}")
    print(f"   2. K_adj Moderado (Média - MG {media_mg:.1f} dB): {K_adj_medio:.3f}")
    print(f"   3. K_adj Agressivo (Melhor Caso - MG {melhor_mg:.1f} dB): {K_adj_otimista:.3f}")

# Plot estruturado dos resultados para inspeção visual
plt.figure(figsize=(10, 5))
for i, tf in enumerate(sys_tfs):
    ct.bode_plot(tf, omega=omega_vetor, dB=True, Hz=False, label=f'Modelo {i+1}', display_margins=True)
plt.legend()
plt.suptitle("Comparação de Resposta de Frequência dos Modelos Carregados")
plt.show()
