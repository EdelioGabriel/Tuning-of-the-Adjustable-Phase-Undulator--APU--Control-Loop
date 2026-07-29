"""
ANÁLISE DE UM ÚNICO BODE

Esse script tem como objetivo analisar um único Bode, carregando os dados de um arquivo CSV específico e estimando a função de transferência correspondente. Ele utiliza a biblioteca `control` para criar objetos de resposta em frequência (FRD) e realizar a otimização para ajustar os parâmetros da função de transferência.

Autor: Edélio Gabriel Magalhães de Jesus
Data: 2024-06-20

Desenvolvido com auxílio de Inteligência Artificial
"""

import control as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import json
import io
import os

# ===============================================================+
# Configurações importantes
# ================================================================

# Nome do arquivo CSV contendo os dados do Bode a ser analisado
BODE_NAME_FILE = './bode_files_PAPU/Id_1_Pos_without_oversampling_kp_10_Tn_0_20.csv'

# Define qual parte do sistema deve ser analisada
"""
O datataset possui informação das seguintes partes:

    Planta ('Process')
    Malha aberta ('Open-Loop')
    Malha fechada ('Close-Loop')

"""
PARTE_DO_SISTEMA = 'Process'

# =========================================================
# Palpites iniciais (todos em termos de frequência natural wn, rad/s)
# =========================================================

freq_natural_polos_comp = [800, 1100]
freq_natural_zeros_comp = [800]
freq_natural_polos_reais = []
freq_natural_zeros_reais = []

# =========================================================
# Número de polos e zeros estimados
# =========================================================

# Configurações de Polos e Zeros:
tem_polo_origem = True   # True para adicionar (1/s) no denominador
tem_zero_origem = False  # True para adicionar (s) no numerador (Derivador)

n_polos_reais = 0        # Quantidade de polos reais livres
n_zeros_reais = 0        # Quantidade de zeros reais livres

n_pares_polos_complexos = 2     # Quantidade de pares de polos complexos livres
n_pares_zeros_complexos = 1     # Quantidade de pares de zeros complexos livres

# =========================================================
# Criação do objeto FRD a partir de um arquivo CSV
# =========================================================

# Função para realizar o tratamento de string e cabeçalho para extração dos dados
def carregar_dados(arquivo):
    with open(arquivo, 'r') as f:
        linhas = [linha.strip().strip('"') for linha in f]

    idx_header = next(i for i, l in enumerate(linhas) if l.startswith('Frequency'))
    texto_dados = '\n'.join(linhas[idx_header:])

    df = pd.read_csv(io.StringIO(texto_dados), sep='\t')
    return df

df = carregar_dados(BODE_NAME_FILE)

# Extração dos dados do Bode
gain = df[f'{PARTE_DO_SISTEMA}-Gain'].to_numpy()    
phase = df[f'{PARTE_DO_SISTEMA}-Phase'].to_numpy()
frequency = df['Frequency'].to_numpy()

# Conversão do ganho de dB para magnitude linear e da fase de graus para radianos
mag_linear = 10 ** (gain / 20)
phase_rad = np.deg2rad(phase)
omega = 2 * np.pi * frequency

# Criação do objeto FRD (Frequency Response Data) usando os dados carregados
sys_frd = ct.frd(
    mag_linear * np.exp(1j * phase_rad),
    omega,
    name='{PARTE_DO_SISTEMA} Bode Data',
)

# =============================================================================
# Ectração e condiguração dos parâmetros
# =============================================================================

# Extração das frequências e da resposta em frequência complexa do objeto FRD
omega = sys_frd.omega  # Frequências em rad/s
frdata = np.squeeze(sys_frd.frdata)  # Resposta em frequência complexa

# Função de Resíduos para Otimização
def residuals(params):
    K = params[0]
    idx = 1
    
    # Extração dos parâmetros dos zeros reais (agora em wn, rad/s)
    wn_z_real = params[idx : idx + n_zeros_reais]
    idx += n_zeros_reais
    
    # Zeros complexos: cada par precisa de [wn_z, zeta_z]
    params_z_comp = params[idx : idx + 2 * n_pares_zeros_complexos]
    idx += 2 * n_pares_zeros_complexos
    
    # Polos reais (agora em wn, rad/s)
    wn_p_real = params[idx : idx + n_polos_reais]
    idx += n_polos_reais
    
    # Polos complexos: cada par precisa de [wn_z, zeta_z]
    params_p_comp = params[idx : idx + 2 * n_pares_polos_complexos]
    
    s = 1j * omega
    
    # --- NUMERADOR ---
    num_val = (s.copy() if tem_zero_origem else np.ones_like(s, dtype=complex)) * K
    for wz in wn_z_real:
        num_val *= (s / wz + 1.0)
        
    for i in range(n_pares_zeros_complexos):
        wn_z = params_z_comp[2 * i]
        zeta_z = params_z_comp[2 * i + 1]
        num_val *= (s**2 / wn_z**2 + 2 * zeta_z * s / wn_z + 1.0) 
        
    # --- DENOMINADOR ---
    den_val = s.copy() if tem_polo_origem else np.ones_like(s, dtype=complex)
    for wp in wn_p_real:
        den_val *= (s / wp + 1.0)
        
    for i in range(n_pares_polos_complexos):
        wn_p = params_p_comp[2 * i]
        zeta_p = params_p_comp[2 * i + 1]
        den_val *= (s**2 / wn_p**2 + 2 * zeta_p * s / wn_p + 1.0)  
        
    H_est = num_val / den_val
    err = H_est - frdata
    return np.concatenate([err.real, err.imag])

# ==========================================================================
# Configuração dos parâmetros iniciais e limites para a otimização
# ==========================================================================

# Montagem do vetor x0
x0 = [1.0]

# Zeros
for _, f in zip(range(n_zeros_reais), freq_natural_zeros_reais):
    x0 += [f]

for _, f in zip(range(n_pares_zeros_complexos), freq_natural_zeros_comp):
    x0 += [f, 0.7]

# Polos
for _, f in zip(range(n_polos_reais), freq_natural_polos_reais):
    x0 += [f]

for _, f in zip(range(n_pares_polos_complexos), freq_natural_polos_comp):
    x0 += [f, 0.7]

# Montagem dos Limites (bounds)
lower_bounds = [-np.inf]
upper_bounds = [ np.inf]

# Zeros Reais (wn_z > 0, rad/s) e Complexos (wn_z > 0, zeta > 0)
for _ in range(n_zeros_reais):
    lower_bounds.append(1.0); upper_bounds.append(20e3)   # wn_z > 0 (rad/s)
for _ in range(n_pares_zeros_complexos):
    lower_bounds.extend([1.0, 0.1])      # wn > 0 (rad/s), zeta > 0
    upper_bounds.extend([2e3, 1.0])       # ajuste o teto conforme sua banda de interesse

# Polos Reais (wn_p > 0, rad/s) e Complexos (wn_p > 0, zeta > 0)
for _ in range(n_polos_reais):
    lower_bounds.append(1.0); upper_bounds.append(20e3)   # wn_p > 0 (rad/s)
for _ in range(n_pares_polos_complexos):
    lower_bounds.extend([1.0, 0.1])
    upper_bounds.extend([2e3, 1.0])

bounds = (lower_bounds, upper_bounds)

# Executa a Otimização
res = opt.least_squares(residuals, x0, bounds=bounds, x_scale='jac')

print("Otimização concluída.")

print(f"Status : {res.status}")
print(f"Mensagem: {res.message}")
print(f"Nº avaliações: {res.nfev}")
print(f"Custo final: {res.cost:.6e}")
print(f"Erro RMS: {np.sqrt(2*res.cost/len(frdata)):.6e}")

K_bode = res.x[0]
idx = 1

# Zeros Reais (wn_z, rad/s)
wn_z_real_opt = res.x[idx : idx + n_zeros_reais]
idx += n_zeros_reais

# Zeros Complexos [wn_z, zeta_z]
params_z_comp_opt = res.x[idx : idx + 2 * n_pares_zeros_complexos]
idx += 2 * n_pares_zeros_complexos

# Polos Reais (wn_p, rad/s)
wn_p_real_opt = res.x[idx : idx + n_polos_reais]
idx += n_polos_reais

# Polos Complexos [wn_p, zeta_p]
params_p_comp_opt = res.x[idx : idx + 2 * n_pares_polos_complexos]

print("\nParâmetros encontrados:")

print(f"K = {K_bode}")

for i in range(n_pares_zeros_complexos):
    wn_z = params_z_comp_opt[2*i]
    zz = params_z_comp_opt[2*i+1]

    print(f"Zero complexo {i+1}")
    print(f"   wn   = {wn_z}")
    print(f"   zeta = {zz}")

for i in range(n_pares_polos_complexos):
    wn_p = params_p_comp_opt[2*i]
    zp = params_p_comp_opt[2*i+1]

    print(f"Polo complexo {i+1}")
    print(f"   wn   = {wn_p}")
    print(f"   zeta = {zp}")

# ==============================================================================
# 2. MONTAGEM DO MODELO ZPK (Zero-Pole-Gain)
# ==============================================================================
zeros_zpk = []
polos_zpk = []

# --- ZEROS REAIS ---
# Zero em s = -wn_z (pois o termo é (s/wn_z + 1))
zeros_zpk.extend([-wz for wz in wn_z_real_opt])

# --- ZEROS COMPLEXOS ---
# Convertendo (wn, zeta) -> raízes do polinômio (s^2/wn^2 + 2*zeta*s/wn + 1)
for i in range(n_pares_zeros_complexos):
    wn_z = params_z_comp_opt[2 * i]
    zz = params_z_comp_opt[2 * i + 1]
    # Raízes da equação quadrática: s = wn*(-zeta ± j*sqrt(1 - zeta^2))
    roots_z = np.roots([1.0, 2 * zz * wn_z, wn_z**2])
    zeros_zpk.extend(roots_z)

# --- POLOS REAIS ---
# Polo em s = -wn_p (pois o termo é (s/wn_p + 1))
polos_zpk.extend([-wp for wp in wn_p_real_opt])

# --- POLOS COMPLEXOS ---
for i in range(n_pares_polos_complexos):
    wn_p = params_p_comp_opt[2 * i]
    zp = params_p_comp_opt[2 * i + 1]
    roots_p = np.roots([1.0, 2 * zp * wn_p, wn_p**2])
    polos_zpk.extend(roots_p)

# Integrador/Diferenciador na Origem
if tem_zero_origem:
    zeros_zpk.append(0.0)

if tem_polo_origem:
    polos_zpk.append(0.0)

# Cálculo do Ganho ZPK (Conversão da forma wn para a forma ZPK)
# Cada termo real (s/wn + 1) = (1/wn) * (s + wn) = (1/wn) * (s - z)
# Cada termo complexo (s^2/wn^2 + 2*zeta*s/wn + 1) = (1/wn^2) * (s - z1)*(s - z2)
# Logo: K_zpk = K_bode * (fator_escala_num / fator_escala_den)
fator_escala_num = (np.prod(1.0 / wn_z_real_opt) if n_zeros_reais > 0 else 1.0) * \
                    (np.prod(1.0 / params_z_comp_opt[0::2]**2) if n_pares_zeros_complexos > 0 else 1.0)

fator_escala_den = (np.prod(1.0 / wn_p_real_opt) if n_polos_reais > 0 else 1.0) * \
                    (np.prod(1.0 / params_p_comp_opt[0::2]**2) if n_pares_polos_complexos > 0 else 1.0)

K_zpk = K_bode * (fator_escala_num / fator_escala_den)

# Objeto ZPK do python-control
sys_zpk = ct.zpk(zeros_zpk, polos_zpk, K_zpk)


# ==============================================================================
# 3. MONTAGEM DA FUNÇÃO DE TRANSFERÊNCIA POLINOMIAL (Num/Den)
# ==============================================================================

# --- NUMERADOR ---
num_poly = np.array([K_bode])

# Multiplica zeros reais: (s / wn_z + 1)
for wz in wn_z_real_opt:
    num_poly = np.convolve(num_poly, [1.0 / wz, 1.0])

# Multiplica zeros complexos: (s^2/wn_z^2 + 2*zeta*s/wn_z + 1)
for i in range(n_pares_zeros_complexos):
    wn_z = params_z_comp_opt[2 * i]
    zz = params_z_comp_opt[2 * i + 1]
    num_poly = np.convolve(num_poly, [1.0 / wn_z**2, 2 * zz / wn_z, 1.0])

if tem_zero_origem:
    num_poly = np.convolve(num_poly, [1.0, 0.0])  # Multiplica por 's'


# --- DENOMINADOR ---
den_poly = np.array([1.0])

# Multiplica polos reais: (s / wn_p + 1)
for wp in wn_p_real_opt:
    den_poly = np.convolve(den_poly, [1.0 / wp, 1.0])

# Multiplica polos complexos: (s^2/wn_p^2 + 2*zeta*s/wn_p + 1)
for i in range(n_pares_polos_complexos):
    wn_p = params_p_comp_opt[2 * i]
    zp = params_p_comp_opt[2 * i + 1]
    den_poly = np.convolve(den_poly, [1.0 / wn_p**2, 2 * zp / wn_p, 1.0])

if tem_polo_origem:
    den_poly = np.convolve(den_poly, [1.0, 0.0])  # Multiplica por 's'

# Objeto Função de Transferência do python-control
sys_tf = ct.tf(num_poly, den_poly)

# ==============================================================================
# SAÍDA DE RESULTADOS
# ==============================================================================
print("=== PARÂMETROS OTIMIZADOS ===")
print(f"K (Ganho Bode) = {K_bode:.4f}")
print(f"Zero na origem: {tem_zero_origem} | Polo na origem: {tem_polo_origem}")

for i, wz in enumerate(wn_z_real_opt, 1):
    print(f"Zero Real {i}:  wn_z = {wz:.2f} rad/s  |  z = {-wz:.2f} rad/s")

for i, wp in enumerate(wn_p_real_opt, 1):
    print(f"Polo Real {i}:  wn_p = {wp:.2f} rad/s  |  p = {-wp:.2f} rad/s")

print("\n=== OBJETOS GERADOS (python-control) ===")
print(sys_tf)


print("\n========== ZEROS ==========")

for z in ct.zeros(sys_tf):
    print(z)

print("\n========== POLOS ==========")

for p in ct.poles(sys_tf):
    print(p)


# =============================================================================
# Exportação da função de transferência para um arquivo JSON
# =============================================================================
num_list = [float(c) for c in sys_tf.num[0][0]]
den_list = [float(c) for c in sys_tf.den[0][0]]

data_export = {
    'num': num_list,
    'den': den_list,
    'dt': sys_tf.dt if sys_tf.dt is not None else 0
}

_, nome_base = os.path.split(BODE_NAME_FILE)
diretorio = 'tfs_json_PAPU'
nome_arquivo = os.path.join(diretorio, f"{PARTE_DO_SISTEMA}_TF_{nome_base}.json")

with open(nome_arquivo, 'w') as f:
    json.dump(data_export, f, indent=4)

print(f'Função de transferência exportada para {nome_arquivo}')

# =============================================================================
# 4. PLOTANDO O BODE COMPARATIVO
# =============================================================================

# -------------------- Plot no domínio dos dados ------------------------------

ct.bode_plot([sys_frd, sys_tf, sys_zpk], omega=omega, dB=True, Hz=True, legend_loc='lower left')
plt.show()

# -------------------- Plot com domínio expandido ------------------------------ 
# Vetor de frequências original (dos dados medidos)
omega_dados = sys_frd.omega

# Vetor estendido para os modelos analíticos (1 década a mais em cada ponta)
omega_min_ext = omega_dados.min() / 1000
omega_max_ext = omega_dados.max() * 10
omega_ext = np.logspace(np.log10(omega_min_ext), np.log10(omega_max_ext), 500)

# Resposta em frequência dos modelos analíticos no range estendido
resp_tf = ct.frequency_response(sys_tf, omega_ext)
resp_zpk = ct.frequency_response(sys_zpk, omega_ext)

# Extração e tratamento das curvas dos modelos (garantindo 1D array)
mag_tf_db = 20 * np.log10(np.abs(resp_tf.frdata[0, 0]))
mag_zpk_db = 20 * np.log10(np.abs(resp_zpk.frdata[0, 0]))

fase_tf_deg = np.rad2deg(np.unwrap(np.angle(resp_tf.frdata[0, 0])))
fase_zpk_deg = np.rad2deg(np.unwrap(np.angle(resp_zpk.frdata[0, 0])))

# Plot manual (2 subplots: magnitude e fase)
fig, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# --- Magnitude ---
ax_mag.semilogx(omega_dados, 20 * np.log10(np.abs(frdata)), 'o', label='Dados (FRD)', markersize=4)
ax_mag.semilogx(omega_ext, mag_tf_db, label='Modelo TF')
ax_mag.semilogx(omega_ext, mag_zpk_db, '--', label='Modelo ZPK')
ax_mag.set_ylabel('Magnitude (dB)')
ax_mag.legend()
ax_mag.grid(True, which='both')

# --- Fase ---
ax_phase.semilogx(omega_dados, np.rad2deg(np.unwrap(np.angle(frdata))), 'o', markersize=4)
ax_phase.semilogx(omega_ext, fase_tf_deg)
ax_phase.semilogx(omega_ext, fase_zpk_deg, '--')
ax_phase.set_ylabel('Fase (graus)')
ax_phase.set_xlabel('Frequência (rad/s)') # Corrigido se a entrada for rad/s
ax_phase.grid(True, which='both')

# --- Ajuste dinâmico dos limites dos eixos (Ylim) ---
# Define margens de segurança (padding) para o gráfico não colar nas bordas
pad_mag = 5  # 5 dB de folga
pad_fase = 15  # 15 graus de folga

ax_mag.set_ylim(min(mag_tf_db.min(), mag_zpk_db.min()) - pad_mag, 
                max(mag_tf_db.max(), mag_zpk_db.max()) + pad_mag)

ax_phase.set_ylim(min(fase_tf_deg.min(), fase_zpk_deg.min()) - pad_fase, 
                  max(fase_tf_deg.max(), fase_zpk_deg.max()) + pad_fase)

plt.tight_layout()
plt.show()

# ---- Plot dos diagramas com as margens -----------
if PARTE_DO_SISTEMA == 'Open-Loop':
    # Calcula as margens do FRD
    gm, pm, wcg, wcp = ct.margin(sys_frd)
    
    # Verifica se os valores são inválidos (NaN) ou infinitos (sem cruzamento)
    if np.isnan(gm) or np.isnan(pm) or np.isinf(gm) or np.isinf(pm):
        print("Margens não encontradas ou infinitas no FRD. Plotando a Função de Transferência estimada...")
        # Correção: Corrigido o aviso usando 'display_margins=True'
        
        ct.bode_plot(sys_tf, omega=omega_ext, dB=True, Hz=False, display_margins=True)
        plt.show()
    else:
        print("Margens válidas encontradas no FRD. Plotando os dados medidos...")
        # Correção: Corrigido o erro de digitação de 'display_margin' para 'display_margins'
        ct.bode_plot(sys_frd, dB=True, Hz=True, display_margins=True)
        plt.show()
