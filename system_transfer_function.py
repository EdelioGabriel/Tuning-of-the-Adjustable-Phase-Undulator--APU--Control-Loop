"""
 ANÁLISE DE UM ÚNICO BODE

Esse script tem como objetivo analisar um único Bode, carregando os dados de um arquivo CSV específico e estimando a função de transferência correspondente. Ele utiliza a biblioteca `control` para criar objetos de resposta em frequência (FRD) e realizar a otimização para ajustar os parâmetros da função de transferência.

Autor: Edélio Gabriel Magalhães de Jesus
Data: 2024-06-20

"""

import control as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import json
import io
import os
import optuna 

# =========================================================
# Criação do objeto FRD a partir de um arquivo CSV
# =========================================================

# Nome do arquivo CSV contendo os dados do Bode a ser analisado
BODE_NAME_FILE = './bode_files_PAPU/Id_1_Vel_NC_kp_586_Tn_15__1.csv'

def carregar_dados(arquivo):
    with open(arquivo, 'r') as f:
        linhas = [linha.strip().strip('"') for linha in f]

    idx_header = next(i for i, l in enumerate(linhas) if l.startswith('Frequency'))
    texto_dados = '\n'.join(linhas[idx_header:])

    df = pd.read_csv(io.StringIO(texto_dados), sep='\t')
    return df

df = carregar_dados(BODE_NAME_FILE)

# Extração dos dados do Bode
gain = df['Process-Gain'].to_numpy()    
phase = df['Process-Phase'].to_numpy()
frequency = df['Frequency'].to_numpy()

# Conversão do ganho de dB para magnitude linear e da fase de graus para radianos
mag_linear = 10 ** (gain / 20)
phase_rad = np.deg2rad(phase)
omega = 2 * np.pi * frequency

# Criação do objeto FRD (Frequency Response Data) usando os dados carregados
sys_frd = ct.frd(
    mag_linear * np.exp(1j * phase_rad),
    omega,
    name='Process Bode Data',
)

# Extração das frequências e da resposta em frequência complexa do objeto FRD
omega = sys_frd.omega  # Frequências em rad/s
fresp = np.squeeze(sys_frd.frdata)  # Resposta em frequência complexa

# Configurações de Polos e Zeros:
tem_polo_origem = True   # True para adicionar (1/s) no denominador
tem_zero_origem = False  # True para adicionar (s) no numerador (Derivador)

n_polos_reais = 0        # Quantidade de polos reais livres
n_zeros_reais = 0        # Quantidade de zeros reais livres

n_polos_complexos = 2     # Quantidade de polos complexos livres
n_zeros_complexos = 1     # Quantidade de zeros complexos livres

# Função de Resíduos para Otimização
def residuals(params):
    K = params[0]
    idx = 1
    
    # Extração dos parâmetros dos zeros reais
    tau_z_real = params[idx : idx + n_zeros_reais]
    idx += n_zeros_reais
    
    # Zeros complexos: cada par precisa de (tau, zeta)
    params_z_comp = params[idx : idx + 2 * n_zeros_complexos]
    idx += 2 * n_zeros_complexos
    
    # Polos reais
    tau_p_real = params[idx : idx + n_polos_reais]
    idx += n_polos_reais
    
    # Polos complexos: cada par precisa de (tau, zeta)
    params_p_comp = params[idx : idx + 2 * n_polos_complexos]
    
    s = 1j * omega
    
    # --- NUMERADOR ---
    num_val = (s.copy() if tem_zero_origem else np.ones_like(s, dtype=complex)) * K
    for tz in tau_z_real:
        num_val *= (tz * s + 1.0)
        
    for i in range(n_zeros_complexos):
        tau_z = params_z_comp[2 * i]
        zeta_z = params_z_comp[2 * i + 1]
        num_val *= (tau_z**2 * s**2 + 2 * zeta_z * tau_z * s + 1.0)
        
    # --- DENOMINADOR ---
    den_val = s.copy() if tem_polo_origem else np.ones_like(s, dtype=complex)
    for tp in tau_p_real:
        den_val *= (tp * s + 1.0)
        
    for i in range(n_polos_complexos):
        tau_p = params_p_comp[2 * i]
        zeta_p = params_p_comp[2 * i + 1]
        den_val *= (tau_p**2 * s**2 + 2 * zeta_p * tau_p * s + 1.0)
        
    H_est = num_val / den_val
    err = H_est - fresp
    return np.concatenate([err.real, err.imag])

# Configuração dos parâmetros iniciais e limites para a otimização
# Montagem do vetor x0
x0 = [1.0]

# Zeros
x0 += [0.0015] * n_zeros_reais
for _ in range(n_zeros_complexos):
    x0 += [0.0015, 0.5]  # [tau_z, zeta_z]

# Polos
x0 += [0.0015] * n_polos_reais
for _ in range(n_polos_complexos):
    x0 += [0.0015, 0.5]  # [tau_p, zeta_p]

# Montagem dos Limites (bounds)
lower_bounds = [-np.inf]
upper_bounds = [ np.inf]

# Zeros Reais e Complexos
for _ in range(n_zeros_reais):
    lower_bounds.append(1e-5); upper_bounds.append(10.0)
for _ in range(n_zeros_complexos):
    lower_bounds.extend([1e-5, 0.01])  # tau > 0, zeta > 0 (estável)
    upper_bounds.extend([10.0, 2.0])

# Polos Reais e Complexos
for _ in range(n_polos_reais):
    lower_bounds.append(1e-5); upper_bounds.append(10.0)
for _ in range(n_polos_complexos):
    lower_bounds.extend([1e-5, 0.01])  # tau > 0, zeta > 0 (estável)
    upper_bounds.extend([10.0, 2.0])

bounds = (lower_bounds, upper_bounds)

# Executa a Otimização
res = opt.least_squares(residuals, x0, bounds=bounds)

K_bode = res.x[0]
idx = 1

# Zeros Reais
tau_z_opt = res.x[idx : idx + n_zeros_reais]
idx += n_zeros_reais

# Zeros Complexos [tau_z, zeta_z]
params_z_comp_opt = res.x[idx : idx + 2 * n_zeros_complexos]
idx += 2 * n_zeros_complexos

# Polos Reais
tau_p_opt = res.x[idx : idx + n_polos_reais]
idx += n_polos_reais

# Polos Complexos [tau_p, zeta_p]
params_p_comp_opt = res.x[idx : idx + 2 * n_polos_complexos]

# ==============================================================================
# 2. MONTAGEM DO MODELO ZPK (Zero-Pole-Gain)
# ==============================================================================
zeros_zpk = []
polos_zpk = []

# --- ZEROS REAIS ---
zeros_zpk.extend([-1.0 / tz for tz in tau_z_opt])

# --- ZEROS COMPLEXOS ---
# Convertendo (tau, zeta) -> raízes do polinômio (tau^2 s^2 + 2*zeta*tau s + 1)
for i in range(n_zeros_complexos):
    tz = params_z_comp_opt[2 * i]
    zz = params_z_comp_opt[2 * i + 1]
    # Raízes da equação quadrática: s = (-zeta ± j*sqrt(1 - zeta^2)) / tau
    roots_z = np.roots([tz**2, 2 * zz * tz, 1.0])
    zeros_zpk.extend(roots_z)

# --- POLOS REAIS ---
polos_zpk.extend([-1.0 / tp for tp in tau_p_opt])

# --- POLOS COMPLEXOS ---
for i in range(n_polos_complexos):
    tp = params_p_comp_opt[2 * i]
    zp = params_p_comp_opt[2 * i + 1]
    roots_p = np.roots([tp**2, 2 * zp * tp, 1.0])
    polos_zpk.extend(roots_p)

# Integrador/Diferenciador na Origem
if tem_zero_origem:
    zeros_zpk.append(0.0)

if tem_polo_origem:
    polos_zpk.append(0.0)

# Cálculo do Ganho ZPK (Conversão de constante de tempo para forma ZPK)
# K_zpk = K_bode * (prod(tau_z) / prod(tau_p))
prod_tau_z = (np.prod(tau_z_opt) if n_zeros_reais > 0 else 1.0) * \
             (np.prod(params_z_comp_opt[0::2]**2) if n_zeros_complexos > 0 else 1.0)

prod_tau_p = (np.prod(tau_p_opt) if n_polos_reais > 0 else 1.0) * \
             (np.prod(params_p_comp_opt[0::2]**2) if n_polos_complexos > 0 else 1.0)

K_zpk = K_bode * (prod_tau_z / prod_tau_p)

# Objeto ZPK do python-control
sys_zpk = ct.zpk(zeros_zpk, polos_zpk, K_zpk)


# ==============================================================================
# 3. MONTAGEM DA FUNÇÃO DE TRANSFERÊNCIA POLINOMIAL (Num/Den)
# ==============================================================================

# --- NUMERADOR ---
num_poly = np.array([K_bode])

# Multiplica zeros reais: (tau_z * s + 1)
for tz in tau_z_opt:
    num_poly = np.convolve(num_poly, [tz, 1.0])

# Multiplica zeros complexos: (tau_z^2 * s^2 + 2*zeta*tau_z * s + 1)
for i in range(n_zeros_complexos):
    tz = params_z_comp_opt[2 * i]
    zz = params_z_comp_opt[2 * i + 1]
    num_poly = np.convolve(num_poly, [tz**2, 2 * zz * tz, 1.0])

if tem_zero_origem:
    num_poly = np.convolve(num_poly, [1.0, 0.0])  # Multiplica por 's'


# --- DENOMINADOR ---
den_poly = np.array([1.0])

# Multiplica polos reais: (tau_p * s + 1)
for tp in tau_p_opt:
    den_poly = np.convolve(den_poly, [tp, 1.0])

# Multiplica polos complexos: (tau_p^2 * s^2 + 2*zeta*tau_p * s + 1)
for i in range(n_polos_complexos):
    tp = params_p_comp_opt[2 * i]
    zp = params_p_comp_opt[2 * i + 1]
    den_poly = np.convolve(den_poly, [tp**2, 2 * zp * tp, 1.0])

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

for i, tz in enumerate(tau_z_opt, 1):
    print(f"Zero Real {i}:  tau_z = {tz:.5f} s  |  z = {-1/tz:.2f} rad/s")

for i, tp in enumerate(tau_p_opt, 1):
    print(f"Polo Real {i}:  tau_p = {tp:.5f} s  |  p = {-1/tp:.2f} rad/s")

print("\n=== OBJETOS GERADOS (python-control) ===")
print(sys_tf)

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
nome_arquivo = os.path.join(diretorio, f"TF_{nome_base}.json")

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

# Vetor estendido para os modelos analíticos (ex: 1 década a mais em cada ponta)
omega_min_ext = omega_dados.min() / 10
omega_max_ext = omega_dados.max() * 10
omega_ext = np.logspace(np.log10(omega_min_ext), np.log10(omega_max_ext), 500)

# Resposta em frequência dos modelos analíticos no range estendido
resp_tf = ct.frequency_response(sys_tf, omega_ext)
resp_zpk = ct.frequency_response(sys_zpk, omega_ext)

# Plot manual (2 subplots: magnitude e fase)
fig, (ax_mag, ax_phase) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))

# --- Magnitude ---
ax_mag.semilogx(omega_dados, 20 * np.log10(np.abs(fresp)), 'o', label='Dados (FRD)', markersize=4)
ax_mag.semilogx(omega_ext, 20 * np.log10(np.abs(resp_tf.fresp[0, 0])), label='Modelo TF')
ax_mag.semilogx(omega_ext, 20 * np.log10(np.abs(resp_zpk.fresp[0, 0])), '--', label='Modelo ZPK')
ax_mag.set_ylabel('Magnitude (dB)')
ax_mag.legend()
ax_mag.grid(True, which='both')

# --- Fase ---
ax_phase.semilogx(omega_dados, np.rad2deg(np.angle(fresp)), 'o', markersize=4)
ax_phase.semilogx(omega_ext, np.rad2deg(np.unwrap(np.angle(resp_tf.fresp[0, 0]))))
ax_phase.semilogx(omega_ext, np.rad2deg(np.unwrap(np.angle(resp_zpk.fresp[0, 0]))), '--')
ax_phase.set_ylabel('Fase (graus)')
ax_phase.set_xlabel('Frequência (Hz)')
ax_phase.grid(True, which='both')
ax_mag.set_ylim(-200, 0)      # em dB, ajuste conforme seu caso
ax_phase.set_ylim(-500, 0)   # em graus
plt.tight_layout()
plt.show()
