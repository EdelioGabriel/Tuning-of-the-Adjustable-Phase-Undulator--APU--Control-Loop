"""
Esse script tem como objetivo realizar a identificação dos parâmetros do sistema de duas massas a partir da função de transferência obtida do Bode Plot. 

Autor: Edélio Gabriel Magalhães de Jesus
Data: 2024-06-10

Esse script foi desenvlvido com auxílio Intelogência Artificial.
"""

import json
import control as ct
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import io

# =========================================================
# Valores reais (medidos / datasheet) — NÃO SÃO SUPOSIÇÕES
# =========================================================
# Inércia do motor, do datasheet, em kg.cm²
J1_real_kgcm2 = 1.2
J1_real_kgm2 = J1_real_kgcm2 * 1e-4  # kg.cm² -> kg.m²  (1 cm² = 1e-4 m²)

# Constante de torque do motor, medida, em N.m/A
Kt_real = 0.22

# Premissa que permanece (não eliminável, mas agora EXPLÍCITA):
# a entrada do modelo identificado (sys_tf) é corrente elétrica, em Ampères.
# É essa premissa que torna H(0) = Kt / (J1 + J2) fisicamente válido.

# =========================================================
# Leitura do arquivo
# =========================================================

NOME_ARQUIVO = "./tfs_json_PAPU/VF_process_TF_Id_1_Vel_NC_kp_586_Tn_15__1.csv.json"

BODE_NAME_FILE = "./bode_files_PAPU/Id_1_Vel_NC_kp_586_Tn_15__1.csv"

try:
    with open(NOME_ARQUIVO, 'r') as arquivo:
        dados = json.load(arquivo)

        sys_tf = ct.tf(dados['num'], dados['den'])
        print(f"Modelo lido do arquivo '{NOME_ARQUIVO}':")
        print(sys_tf)

except FileNotFoundError:
    raise FileNotFoundError(f"Arquivo '{NOME_ARQUIVO}' não encontrado.")

print(ct.minreal(sys_tf, verbose=True))

# =========================================================
# Extração de polos/zeros complexos do modelo identificado (sys_tf)
# =========================================================

# --- EXTRAÇÃO DE POLOS E ZEROS ---
zeros = ct.zeros(sys_tf)
polos = ct.poles(sys_tf)

# Separa complexos de reais 
zeros_complexos = [z for z in zeros if np.imag(z)]
polos_complexos = [p for p in polos if np.imag(p)]

# Polos e zeros reais
polos_reais = [p for p in polos if np.abs(np.imag(p)) <= 1e-15 and not np.isclose(p, 0.0, atol=1e-5)]
zeros_reais = [z for z in zeros if np.abs(np.imag(z)) <= 1e-15 and not np.isclose(z, 0.0, atol=1e-5)]

# --- VALIDAÇÃO FÍSICA SEGUNDO A ESTRUTURA GERADA ---
if len(polos_complexos) >= 2:
    # Seleciona o primeiro par conjugado para extrair a frequência natural
    p_comp = polos_complexos[0]
    omega_p = np.abs(p_comp)
    zeta_p = -np.real(p_comp) / omega_p
    print(f"\n[OK] Polos complexos: ω_p = {omega_p:.2f} rad/s, ζ_p = {zeta_p:.3f}")
else:
    raise ValueError("O modelo gerado não possui o par de polos complexos esperado para a dinâmica de duas massas.")

# Tratamento caso o modelo não possua zeros complexos
if len(zeros_complexos) >= 2:
    z_comp = zeros_complexos[0]
    omega_z = np.abs(z_comp)
    zeta_z = -np.real(z_comp) / omega_z
    print(f"[OK] Zeros complexos: ω_z = {omega_z:.2f} rad/s, ζ_z = {zeta_z:.3f}")
else:
    print("\n[Aviso] O modelo atual não estimou zeros complexos (acoplamento infinitamente rígido).")
    # Para evitar quebrar o cálculo físico de J1 e J2 mais abaixo, define-se um limite analítico
    omega_z = np.inf 


# Calcular frequências naturais e amortecimentos para todos os pares
w_z_lista = [np.abs(z) for z in zeros_complexos]
w_p_lista = [np.abs(p) for p in polos_complexos]
zeta_z_lista = [-np.real(z) / np.abs(z) for z in zeros_complexos]
zeta_p_lista = [-np.real(p) / np.abs(p) for p in polos_complexos]

# Usar a média 
w_z = np.mean(w_z_lista)
zeta_z = np.mean(zeta_z_lista)

# Escolher frequencia mais alta para o modo torsinal
indice = np.argmax(w_p_lista)

w_p = w_p_lista[indice]
zeta_p = zeta_p_lista[indice]

print(f"\nZeros complexos encontrados: {len(zeros_complexos)}")
for i, z in enumerate(zeros_complexos):
    print(f"  z[{i}] = {z:.6f} → ω_z = {np.abs(z):.6f}, ζ = {zeta_z_lista[i]:.6f}")
print(f"Média: ω_z = {w_z:.6f}, ζ_z = {zeta_z:.6f}")

print(f"\nPolos complexos encontrados: {len(polos_complexos)}")
for i, p in enumerate(polos_complexos):
    print(f"  p[{i}] = {p:.6f} → ω_p = {np.abs(p):.6f}, ζ = {zeta_p_lista[i]:.6f}")
print(f"Média: ω_p = {w_p:.6f}, ζ_p = {zeta_p:.6f}")

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

# Conversão do ganho de dB para magnitude linear
mag_linear = 10 ** (gain / 20)
phase_rad = np.deg2rad(phase)
omega = 2 * np.pi * frequency

# Extrair ganho DC direto da FT 
"""
H(0) = numerador(s=0) / denominador(s=0)
"""
num_coef = sys_tf.num[0][0] 
den_coef = sys_tf.den[0][0]  

# =========================================================
# Ganho DC
# =========================================================

print("\n--- EXTRAÇÃO DO GANHO DC ---")

# Operador s
s = ct.tf([1, 0], [1])

# Verifica se existe polo na origem
try:
    H_0 = ct.dcgain(sys_tf)

    if np.isinf(H_0):
        print(H_0)
        print("Polo na origem detectado.")
        print("Calculando ganho DC da FT de velocidade (s·G(s)).")

        sys_vel = s * sys_tf
        sys_vel = ct.minreal(sys_vel, verbose=True)

        H_0 = ct.dcgain(sys_vel)
        print(H_0)
    else:
        print("Modelo sem polo na origem.")

except Exception as e:
    raise RuntimeError(f"Erro ao calcular o ganho DC: {e}")

print(f"Ganho DC = {H_0:.6e}")

# =========================================================
# J_total a partir de valores REAIS (Kt medido), não suposição
# =========================================================
"""
H(0) = Kt / (J1 + J2)   [entrada = corrente, em A]
Logo: J1 + J2 = Kt / H(0)
"""
J_total = Kt_real / H_0

print(f"\nRelação: H(0) = Kt / (J₁ + J₂), com Kt = {Kt_real} N.m/A (medido)")
print(f"J₁ + J₂ = {J_total:.6e} kg.m²")

# J2 obtido diretamente, usando J1 real (datasheet) — não há mais incógnita
# a resolver via estrutura de zeros/polos.
J2_real = J_total - J1_real_kgm2

print(f"\n--- PARÂMETROS IDENTIFICADOS (COM J1 e Kt REAIS) ---")
print(f"J1 (Motor, datasheet) : {J1_real_kgm2:.6e} kg.m²")
print(f"J2 (Inércia Ext, calc): {J2_real:.6e} kg.m²")

if J2_real <= 0:
    print("⚠ ATENÇÃO: J2 calculado é <= 0. Isso indica inconsistência entre")
    print("   H(0), Kt_real e J1_real_kgm2 — reveja a premissa de entrada em corrente")
    print("   ou a qualidade do ajuste de sys_tf antes de prosseguir.")

# Rigidez Torcional e Amortecimento, a partir do par de zeros complexos
K_theta_calibrado = J2_real * (w_z**2)
D_theta_calibrado = 2 * zeta_z * J2_real * w_z

print(f"K_theta (Rigidez)     : {K_theta_calibrado:.6f} N.m/rad")
print(f"D_theta (Amort.)      : {D_theta_calibrado:.6e} N.m.s/rad")

# =========================================================
# Checagem de consistência (diagnóstico, NÃO usado no cálculo acima)
# =========================================================
# A razão J2/J1 prevista pela estrutura ideal de duas massas é:
#   J2/J1 = (ω_p/ω_z)² - 1
# Comparamos com a razão obtida a partir de J1 e J2 reais. Se divergirem
# muito, é sinal de dinâmica extra não capturada pelo modelo ideal
# (backlash, zeros elétricos, etc. — ver discussões anteriores).
razao_J_teorica = (w_p / w_z)**2 - 1.0
razao_J_real = J2_real / J1_real_kgm2

print(f"\n--- CHECAGEM DE CONSISTÊNCIA (diagnóstico) ---")
print(f"J₂/J₁ previsto pela estrutura ideal (ω_p/ω_z)² - 1: {razao_J_teorica:.4f}")
print(f"J₂/J₁ obtido com J1 e Kt reais                    : {razao_J_real:.4f}")
desvio_pct = abs(razao_J_teorica - razao_J_real) / razao_J_real * 100
print(f"Desvio relativo entre as duas razões: {desvio_pct:.1f}%")
if desvio_pct > 20:
    print("Desvio significativo: o modelo ideal de duas massas pode não")
    print("capturar toda a dinâmica observada (ver zeros extras, backlash, etc.)")

num_fisico = [J2_real, D_theta_calibrado, K_theta_calibrado]
den_fisico = [J1_real_kgm2 * J2_real, (J1_real_kgm2 + J2_real) *
            D_theta_calibrado, (J1_real_kgm2 + J2_real) * K_theta_calibrado, 0.0]
# Multiplicando por Kt_real, já que a entrada real do motor é corrente (A),
# não torque direto — assim sys_fisico fica na mesma base de sys_tf/sys_frd
num_fisico = [Kt_real * c for c in num_fisico]
sys_fisico = ct.tf(num_fisico, den_fisico)

# Criação do objeto FRD (Frequency Response Data)
sys_frd = ct.frd(
    mag_linear * np.exp(1j * phase_rad),
    omega,
    name='Open-Loop Bode Data',
)

# Plotar a comparação com o modelo otimizado pelo least_squares (sys_tf)
plt.figure(figsize=(10, 6))
ct.bode_plot(
    [sys_frd,sys_tf, sys_fisico],
    omega=omega,
    dB=True,
    Hz=True,
    label=['Curva original',
           'Modelo Ajustado (Least Squares)', 
           'Modelo Físico'],
    legend_loc='lower left'
)

plt.suptitle('Validação: Modelo Ajustado vs Modelo Físico Teórico', fontsize=12)
plt.show()