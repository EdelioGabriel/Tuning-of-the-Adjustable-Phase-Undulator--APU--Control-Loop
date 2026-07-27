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
# Valorres reais
# ========================================================
J1_twin_gcm2 = None  # g·cm²

if J1_twin_gcm2 is not None:
    J1_twin_kgm2 = J1_twin_gcm2 * 1e-7  # Converter para kg·m²

# =========================================================
# Leitura do arquivo
# =========================================================

NOME_ARQUIVO = "TF_Id_1_Vel_NC_kp_586_Tn_15__1.csv.json"

BODE_NAME_FILE = "Id_1_Vel_NC_kp_586_Tn_15__1.csv"

try:
    with open(NOME_ARQUIVO, 'r') as arquivo:
        dados = json.load(arquivo)

        sys_tf = ct.tf(dados['num'], dados['den'])
        print(f"Modelo lido do arquivo '{NOME_ARQUIVO}':")
        print(sys_tf)

except FileNotFoundError:
    raise FileNotFoundError(f"Arquivo '{NOME_ARQUIVO}' não encontrado.")

# =========================================================
# Cálculo dos parâmetros (SEM Kt, identificando J1)
# =========================================================

# --- EXTRAÇÃO ROBUSTA DE POLOS E ZEROS ---
zeros = ct.zeros(sys_tf)
polos = ct.poles(sys_tf)

# Separa complexos de reais (filtrando o polo na origem em s=0)
zeros_complexos = [z for z in zeros if np.abs(np.imag(z)) > 1e-10]
polos_complexos = [p for p in polos if np.abs(np.imag(p)) > 1e-10]

# Polos e zeros reais (excluindo o zero se estiver na origem)
polos_reais = [p for p in polos if np.abs(np.imag(p)) <= 1e-10 and not np.isclose(p, 0.0, atol=1e-5)]
zeros_reais = [z for z in zeros if np.abs(np.imag(z)) <= 1e-10 and not np.isclose(z, 0.0, atol=1e-5)]

# --- VALIDAÇÃO FÍSICA SEGUNDO A ESTRUTURA GERADA ---
if len(polos_complexos) >= 2:
    # Seleciona o primeiro par conjugado para extrair a frequência natural
    p_comp = polos_complexos[0]
    omega_p = np.abs(p_comp)
    zeta_p = -np.real(p_comp) / omega_p
    print(f"\n[OK] Polos complexos: ω_p = {omega_p:.2f} rad/s, ζ_p = {zeta_p:.3f}")
else:
    raise ValueError("O modelo gerado não possui o par de polos complexos esperado para a dinâmica de duas massas.")

# Tratamento caso o modelo não possua zeros complexos (Como a FT atual de 3ª ordem)
if len(zeros_complexos) >= 2:
    z_comp = zeros_complexos[0]
    omega_z = np.abs(z_comp)
    zeta_z = -np.real(z_comp) / omega_z
    print(f"[OK] Zeros complexos: ω_z = {omega_z:.2f} rad/s, ζ_z = {zeta_z:.3f}")
else:
    print("\n[Aviso] O modelo atual não estimou zeros complexos (acoplamento infinitamente rígido).")
    # Para evitar quebrar o cálculo físico de J1 e J2 mais abaixo, define-se um limite analítico
    # Ou você pode forçar o Optuna a buscar uma ordem que obrigatoriamente tenha zeros complexos.
    omega_z = np.inf 


# Calcular frequências naturais e amortecimentos para todos os pares
w_z_lista = [np.abs(z) for z in zeros_complexos]
w_p_lista = [np.abs(p) for p in polos_complexos]
zeta_z_lista = [-np.real(z) / np.abs(z) for z in zeros_complexos]
zeta_p_lista = [-np.real(p) / np.abs(p) for p in polos_complexos]

# Usar a média 
w_z = np.mean(w_z_lista)
w_p = np.mean(w_p_lista)
zeta_z = np.mean(zeta_z_lista)
zeta_p = np.mean(zeta_p_lista)

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

H_0 = num_coef[-1] / den_coef[-1]  

print(f"\n--- EXTRAÇÃO DO GANHO DC DA FT ---")
print(f"Numerador em s=0: {num_coef[-1]:.6e}")
print(f"Denominador em s=0: {den_coef[-1]:.6e}")
print(f"Ganho DC (H(0)): {H_0:.6e}")

# Relação teórica (com entrada = torque, Kt implícito = 1):
"""
H(0) = 1 / (J1 + J2)
# Logo: J1 + J2 = 1 / H(0)
"""
J_total = 1.0 / H_0 if H_0 > 0 else 1.0

print(f"\nRelação: H(0) = 1/(J₁ + J₂)")
print(f"J₁ + J₂ = {J_total:.6e} kg.m²")

# Razão entre inércias a partir dos zeros/polos
"""
J2/J1 = (ω_p/ω_z)^2 - 1
"""
razao_J = (w_p / w_z)**2 - 1.0

print(f"\nRazão: J₂/J₁ = (ω_p/ω_z)² - 1 = {razao_J:.6f}")

# Resolver sistema linear:
"""
J1 + J2 = J_total
J2 = razao_J * J1

Substituindo:

J1 + razao_J * J1 = J_total
J1 * (1 + razao_J) = J_total
"""

J1_real = J_total / (1.0 + razao_J)
J2_calibrado = J_total - J1_real

# Rigidez Torcional
K_theta_calibrado = J2_calibrado * (w_z**2)

# Amortecimento
D_theta_calibrado = 2 * zeta_z * J2_calibrado * w_z

print(f"\n--- PARÂMETROS IDENTIFICADOS (SEM USAR Kt NEM DATASHEET) ---")
print(f"J1 (Motor)       : {J1_real:.6e} kg.m²")
print(f"J2 (Inércia Ext) : {J2_calibrado:.6e} kg.m²")
print(f"K_theta (Rigidez): {K_theta_calibrado:.6f} N.m/rad")
print(f"D_theta (Amort.) : {D_theta_calibrado:.6e} N.m.s/rad")

# =========================================================
# Comparação com valor no TwinCAT
# =========================================================

if J1_twin_gcm2:
    print(f"\n--- COMPARAÇÃO COM DATASHEET ---")
    print(
        f"J1 (Datasheet)   : {J1_twin_gcm2} g·cm² = {J1_twin_kgm2:.6e} kg.m²")
    print(f"J1 (Identificado): {J1_real:.6e} kg.m²")
    print(f"Razão (Identificado / Datasheet): {J1_real / J1_twin_kgm2:.2f}x")
    print(f"Erro absoluto: {abs(J1_real - J1_twin_kgm2):.6e} kg.m²")
    print(
        f"Erro relativo: {abs(J1_real - J1_twin_kgm2) / J1_twin_kgm2 * 100:.2f}%")

    # Análise: Se a razão é grande, significa que há Kt implícito
    if J1_real / J1_twin_kgm2 > 10:
        print(
            f"\n⚠ ATENÇÃO: J1 identificado é {J1_real / J1_twin_kgm2:.0f}x maior que o datasheet!")
        print(f"   Possível causa: Kt ≠ 1")

        # Se a entrada é corrente (não torque), então:
        """
        H(0) = Kt / (J1 + J2)
        Logo: Kt = H(0) * (J1_real + J2_real)
        """
        Kt_implicito = H_0 * (J1_real + J2_calibrado)
        print(f"   Kt implícito (H(0) × J_total): {Kt_implicito:.6f}")

        # Recalcular com J1 do datasheet
        print(f"\n   Recalculando com J1 = datasheet e Kt implícito:")
        J2_recalc = J2_calibrado  # Mantém a razão

        # Se entrada é corrente: 
        """
        H(0) = Kt / (J1 + J2)
        Kt = H(0) * (J1_twin_kgm2 + J2_recalc)
        """
        Kt_correto = H_0 * (J1_twin_kgm2 + J2_calibrado)
        print(f"   K_t (estimado): {Kt_correto:.6f}")
        print(f"   J1 (datasheet): {J1_twin_kgm2:.6e} kg.m²")
        print(f"   J2 (recalc): {J2_calibrado:.6e} kg.m²")
else:
    pass

num_fisico = [J2_calibrado, D_theta_calibrado, K_theta_calibrado]
den_fisico = [J1_real * J2_calibrado, (J1_real + J2_calibrado) *
            D_theta_calibrado, (J1_real + J2_calibrado) * K_theta_calibrado, 0.0]
# SEM multiplicar por Kt (entrada é torque direto)
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
    [sys_frd, sys_tf],
    omega=omega,
    dB=True,
    Hz=True,
    label=['Curva original',
           'Modelo Ajustado (Least Squares)'],
    legend_loc='lower left'
)

plt.suptitle('Validação: Modelo Ajustado vs Modelo Físico Teórico', fontsize=12)
plt.show()
