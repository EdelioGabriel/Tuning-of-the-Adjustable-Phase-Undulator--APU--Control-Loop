"""
ANÁLISE DE UM ÚNICO BODE — BUSCA DE ESTRUTURA (OPTUNA) + PARAMETRIZAÇÃO EM wn/zeta

Versão que combina:
  - a busca automática de estrutura (Optuna + AICc) do script `optuna_transfer_function_analysis.py`
  - a parametrização física em frequência natural (wn, rad/s) e amortecimento (zeta) do
    script `system_transfer_function.py`

Por que mudar de tau para wn:
  Termos em tau: (tau*s + 1)  e  (tau^2 s^2 + 2*zeta*tau*s + 1)
  Termos em wn:  (s/wn + 1)   e  (s^2/wn^2 + 2*zeta*s/wn + 1)
  Fisicamente equivalentes (tau = 1/wn), mas wn é a grandeza que você lê direto do Bode
  (posição do pico de ressonância em rad/s), então os bounds e o x0 podem ser expressos
  na mesma escala dos dados medidos -- isso é o que torna a otimização mais robusta.

Estratégia de x0 (chute inicial):
  No script de referência, os chutes (ex: 400, 1000 rad/s) foram escolhidos manualmente
  olhando o Bode. Aqui, como o Optuna testa uma estrutura diferente a cada trial, não dá
  pra fixar esses valores -- em vez disso, os chutes de wn para cada polo/zero são
  espalhados logaritmicamente ao longo da faixa de frequência dos próprios dados
  (omega.min() a omega.max()). Isso substitui o x0 fixo em 0.01 (na forma tau) que causava
  convergência inconsistente entre variantes estruturais.

Autor: Edélio Gabriel Magalhães de Jesus
Desenvolvido com auxílio de Inteligência Artificial
"""

import control as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import json
import optuna
import io
import os

# =========================================================
# Configurações importantes
# =========================================================

BODE_NAME_FILE = './bode_files_PAPU/Id_1_Pos_without_oversampling_kp_10_Tn_0_20.csv'
PARTE_DO_SISTEMA = 'Process'   # 'Process', 'Open-Loop' ou 'Close-Loop'

N_TRIALS = 500

# Bounds de frequência natural (rad/s) e de amortecimento (zeta)
# Por padrão, deixamos a busca cobrir uma década abaixo e uma acima da faixa medida
WN_MARGEM_DECADAS = 0.3

# =========================================================
# Criação do objeto FRD a partir de um arquivo CSV
# =========================================================

def carregar_dados(arquivo):
    with open(arquivo, 'r') as f:
        linhas = [linha.strip().strip('"') for linha in f]

    idx_header = next(i for i, l in enumerate(linhas) if l.startswith('Frequency'))
    texto_dados = '\n'.join(linhas[idx_header:])

    df = pd.read_csv(io.StringIO(texto_dados), sep='\t')
    return df

df = carregar_dados(BODE_NAME_FILE)

gain = df[f'{PARTE_DO_SISTEMA}-Gain'].to_numpy()
phase = df[f'{PARTE_DO_SISTEMA}-Phase'].to_numpy()
frequency = df['Frequency'].to_numpy()

mag_linear = 10 ** (gain / 20)
phase_rad = np.deg2rad(phase)
omega_data = 2 * np.pi * frequency

sys_frd = ct.frd(
    mag_linear * np.exp(1j * phase_rad),
    omega_data,
    name=f'{PARTE_DO_SISTEMA} Bode Data',
)

omega = sys_frd.omega
fresp = np.squeeze(sys_frd.frdata)

N_amostras = len(omega)

# Faixa de frequência natural coberta pela busca (em rad/s)
WN_MIN = omega.min() / (10 ** WN_MARGEM_DECADAS)
WN_MAX = omega.max() * (10 ** WN_MARGEM_DECADAS)

# ==============================================================================
# 1. FUNÇÕES AUXILIARES (residuals, x0, bounds) — PARAMETRIZAÇÃO EM wn/zeta
# ==============================================================================

def make_residuals(n_zeros_reais, n_zeros_complexos, n_polos_reais, n_polos_complexos,
                    tem_zero_origem, tem_polo_origem):
    """Fecha (closure) sobre omega e fresp e retorna a função de resíduos
    para a estrutura de polos/zeros especificada, parametrizada em wn/zeta."""

    def residuals(params):
        K = params[0]
        idx = 1

        wn_z_real = params[idx: idx + n_zeros_reais]
        idx += n_zeros_reais

        params_z_comp = params[idx: idx + 2 * n_zeros_complexos]
        idx += 2 * n_zeros_complexos

        wn_p_real = params[idx: idx + n_polos_reais]
        idx += n_polos_reais

        params_p_comp = params[idx: idx + 2 * n_polos_complexos]

        s = 1j * omega

        # --- NUMERADOR ---
        num_val = (s.copy() if tem_zero_origem else np.ones_like(s, dtype=complex)) * K
        for wz in wn_z_real:
            num_val *= (s / wz + 1.0)

        for i in range(n_zeros_complexos):
            wn_z = params_z_comp[2 * i]
            zeta_z = params_z_comp[2 * i + 1]
            num_val *= (s**2 / wn_z**2 + 2 * zeta_z * s / wn_z + 1.0)

        # --- DENOMINADOR ---
        den_val = s.copy() if tem_polo_origem else np.ones_like(s, dtype=complex)
        for wp in wn_p_real:
            den_val *= (s / wp + 1.0)

        for i in range(n_polos_complexos):
            wn_p = params_p_comp[2 * i]
            zeta_p = params_p_comp[2 * i + 1]
            den_val *= (s**2 / wn_p**2 + 2 * zeta_p * s / wn_p + 1.0)

        H_est = num_val / den_val
        err = H_est - fresp
        return np.concatenate([err.real, err.imag])

    return residuals


def make_x0_bounds(n_zeros_reais, n_zeros_complexos, n_polos_reais, n_polos_complexos):
    """Monta x0 e bounds na forma wn/zeta.

    Os chutes de wn (para cada polo/zero, real ou complexo) são espalhados
    logaritmicamente entre WN_MIN e WN_MAX, na ordem: zeros reais, zeros
    complexos, polos reais, polos complexos. Isso evita que todos os
    parâmetros comecem no mesmo ponto (o que gerava mínimos locais
    diferentes a cada estrutura testada).
    """

    n_total_wn = n_zeros_reais + n_zeros_complexos + n_polos_reais + n_polos_complexos

    if n_total_wn > 0:
        # Espalha os chutes log-espaçados dentro da faixa observada nos dados
        wn_guesses = np.logspace(np.log10(omega.min()), np.log10(omega.max()), n_total_wn)
    else:
        wn_guesses = np.array([])

    guess_iter = iter(wn_guesses)

    x0 = [1.0]
    lower_bounds = [-np.inf]
    upper_bounds = [np.inf]

    # Zeros reais
    for _ in range(n_zeros_reais):
        x0 += [next(guess_iter)]
        lower_bounds.append(WN_MIN); upper_bounds.append(WN_MAX)

    # Zeros complexos [wn_z, zeta_z]
    for _ in range(n_zeros_complexos):
        x0 += [next(guess_iter), 0.5]
        lower_bounds.extend([WN_MIN, 0.01]); upper_bounds.extend([WN_MAX, 1.0])

    # Polos reais
    for _ in range(n_polos_reais):
        x0 += [next(guess_iter)]
        lower_bounds.append(WN_MIN); upper_bounds.append(WN_MAX)

    # Polos complexos [wn_p, zeta_p]
    for _ in range(n_polos_complexos):
        x0 += [next(guess_iter), 0.5]
        lower_bounds.extend([WN_MIN, 0.01]); upper_bounds.extend([WN_MAX, 2.0])

    bounds = (lower_bounds, upper_bounds)
    return x0, bounds


# ==============================================================================
# 2. BUSCA DA ESTRUTURA (Optuna) — decide quantos polos/zeros usar
# ==============================================================================

def objective(trial):
    n_polos_reais = trial.suggest_int('n_polos_reais', 0, 2)
    n_zeros_reais = trial.suggest_int('n_zeros_reais', 0, 2)
    n_polos_complexos = trial.suggest_int('n_polos_complexos', 0, 2)
    n_zeros_complexos = trial.suggest_int('n_zeros_complexos', 0, 2)
    tem_polo_origem = trial.suggest_categorical('tem_polo_origem', [True, False])
    tem_zero_origem = trial.suggest_categorical('tem_zero_origem', [True, False])

    grau_num = n_zeros_reais + 2 * n_zeros_complexos + (1 if tem_zero_origem else 0)
    grau_den = n_polos_reais + 2 * n_polos_complexos + (1 if tem_polo_origem else 0)

    if grau_num > grau_den or grau_den == 0:
        raise optuna.exceptions.TrialPruned()

    residuals = make_residuals(
        n_zeros_reais, n_zeros_complexos, n_polos_reais, n_polos_complexos,
        tem_zero_origem, tem_polo_origem
    )
    x0, bounds = make_x0_bounds(n_zeros_reais, n_zeros_complexos, n_polos_reais, n_polos_complexos)

    K_params = len(x0)

    if N_amostras - K_params - 1 <= 0:
        raise optuna.exceptions.TrialPruned()

    try:
        res = opt.least_squares(residuals, x0, bounds=bounds, x_scale='jac', max_nfev=3000)
    except Exception:
        raise optuna.exceptions.TrialPruned()

    custo_rss = np.sum(res.fun ** 2)

    if custo_rss <= 1e-12:
        custo_rss = 1e-12

    aic = N_amostras * np.log(custo_rss / N_amostras) + 2 * K_params
    aicc = aic + (2 * K_params * (K_params + 1)) / (N_amostras - K_params - 1)

    return aicc


sampler = optuna.samplers.TPESampler(seed=367)
study = optuna.create_study(
    direction='minimize',
    sampler=sampler,
    study_name='polos_zeros_aicc_wn'
)

print("Buscando a estrutura ideal de polos e zeros baseada no AICc (parametrização wn/zeta)...")
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True, n_jobs=-1)

print("\nMelhores hiperparâmetros (estrutura):")
for chave, valor in study.best_params.items():
    print(f"  {chave}: {valor}")
print(f"\nMelhor score: {study.best_value:.6f}")


# ==============================================================================
# 3. FIT FINAL: refina os parâmetros contínuos usando a MELHOR estrutura encontrada
# ==============================================================================

best = study.best_params
n_polos_reais = best['n_polos_reais']
n_zeros_reais = best['n_zeros_reais']
n_polos_complexos = best['n_polos_complexos']
n_zeros_complexos = best['n_zeros_complexos']
tem_polo_origem = best['tem_polo_origem']
tem_zero_origem = best['tem_zero_origem']

residuals = make_residuals(
    n_zeros_reais, n_zeros_complexos, n_polos_reais, n_polos_complexos,
    tem_zero_origem, tem_polo_origem
)
x0, bounds = make_x0_bounds(n_zeros_reais, n_zeros_complexos, n_polos_reais, n_polos_complexos)

res = opt.least_squares(residuals, x0, bounds=bounds, x_scale='jac', max_nfev=20000, xtol=1e-12, ftol=1e-12)

print("\nOtimização final concluída.")
print(f"Status : {res.status}")
print(f"Mensagem: {res.message}")
print(f"Nº avaliações: {res.nfev}")
print(f"Custo final: {res.cost:.6e}")
print(f"Erro RMS: {np.sqrt(2 * res.cost / len(fresp)):.6e}")

K_bode = res.x[0]
idx = 1

wn_z_real_opt = res.x[idx: idx + n_zeros_reais]
idx += n_zeros_reais

params_z_comp_opt = res.x[idx: idx + 2 * n_zeros_complexos]
idx += 2 * n_zeros_complexos

wn_p_real_opt = res.x[idx: idx + n_polos_reais]
idx += n_polos_reais

params_p_comp_opt = res.x[idx: idx + 2 * n_polos_complexos]

# ==============================================================================
# 4. MONTAGEM DO MODELO ZPK (Zero-Pole-Gain) — forma wn
# ==============================================================================
zeros_zpk = []
polos_zpk = []

# --- ZEROS REAIS --- (zero em s = -wn, pois o termo é (s/wn + 1))
zeros_zpk.extend([-wz for wz in wn_z_real_opt])

# --- ZEROS COMPLEXOS --- (s^2/wn^2 + 2*zeta*s/wn + 1)
for i in range(n_zeros_complexos):
    wn_z = params_z_comp_opt[2 * i]
    zz = params_z_comp_opt[2 * i + 1]
    roots_z = np.roots([1.0, 2 * zz * wn_z, wn_z**2])
    zeros_zpk.extend(roots_z)

# --- POLOS REAIS ---
polos_zpk.extend([-wp for wp in wn_p_real_opt])

# --- POLOS COMPLEXOS ---
for i in range(n_polos_complexos):
    wn_p = params_p_comp_opt[2 * i]
    zp = params_p_comp_opt[2 * i + 1]
    roots_p = np.roots([1.0, 2 * zp * wn_p, wn_p**2])
    polos_zpk.extend(roots_p)

if tem_zero_origem:
    zeros_zpk.append(0.0)

if tem_polo_origem:
    polos_zpk.append(0.0)

# Ganho ZPK: cada termo real (s/wn + 1) = (1/wn)*(s + wn); cada termo complexo
# (s^2/wn^2 + 2*zeta*s/wn + 1) = (1/wn^2)*(s - z1)*(s - z2)
fator_escala_num = (np.prod(1.0 / wn_z_real_opt) if n_zeros_reais > 0 else 1.0) * \
                   (np.prod(1.0 / params_z_comp_opt[0::2] ** 2) if n_zeros_complexos > 0 else 1.0)

fator_escala_den = (np.prod(1.0 / wn_p_real_opt) if n_polos_reais > 0 else 1.0) * \
                   (np.prod(1.0 / params_p_comp_opt[0::2] ** 2) if n_polos_complexos > 0 else 1.0)

K_zpk = K_bode * (fator_escala_num / fator_escala_den)

sys_zpk = ct.zpk(zeros_zpk, polos_zpk, K_zpk)


# ==============================================================================
# 5. MONTAGEM DA FUNÇÃO DE TRANSFERÊNCIA POLINOMIAL (Num/Den) — forma wn
# ==============================================================================

num_poly = np.array([K_bode])

for wz in wn_z_real_opt:
    num_poly = np.convolve(num_poly, [1.0 / wz, 1.0])

for i in range(n_zeros_complexos):
    wn_z = params_z_comp_opt[2 * i]
    zz = params_z_comp_opt[2 * i + 1]
    num_poly = np.convolve(num_poly, [1.0 / wn_z**2, 2 * zz / wn_z, 1.0])

if tem_zero_origem:
    num_poly = np.convolve(num_poly, [1.0, 0.0])

den_poly = np.array([1.0])

for wp in wn_p_real_opt:
    den_poly = np.convolve(den_poly, [1.0 / wp, 1.0])

for i in range(n_polos_complexos):
    wn_p = params_p_comp_opt[2 * i]
    zp = params_p_comp_opt[2 * i + 1]
    den_poly = np.convolve(den_poly, [1.0 / wn_p**2, 2 * zp / wn_p, 1.0])

if tem_polo_origem:
    den_poly = np.convolve(den_poly, [1.0, 0.0])

sys_tf = ct.tf(num_poly, den_poly)

# ==============================================================================
# SAÍDA DE RESULTADOS
# ==============================================================================
print("\n=== ESTRUTURA ESCOLHIDA PELO OPTUNA ===")
print(f"n_zeros_reais={n_zeros_reais}, n_zeros_complexos={n_zeros_complexos}, "
      f"n_polos_reais={n_polos_reais}, n_polos_complexos={n_polos_complexos}, "
      f"tem_zero_origem={tem_zero_origem}, tem_polo_origem={tem_polo_origem}")

print("\n=== PARÂMETROS OTIMIZADOS (FIT FINAL, forma wn/zeta) ===")
print(f"K (Ganho Bode) = {K_bode:.4f}")

for i, wz in enumerate(wn_z_real_opt, 1):
    print(f"Zero Real {i}:  wn_z = {wz:.2f} rad/s  |  z = {-wz:.2f} rad/s")

for i in range(n_zeros_complexos):
    wn_z = params_z_comp_opt[2 * i]
    zz = params_z_comp_opt[2 * i + 1]
    print(f"Par de Zeros Complexos {i+1}: wn = {wn_z:.2f} rad/s ({wn_z/(2*np.pi):.2f} Hz)  |  zeta = {zz:.3f}")

for i, wp in enumerate(wn_p_real_opt, 1):
    print(f"Polo Real {i}:  wn_p = {wp:.2f} rad/s  |  p = {-wp:.2f} rad/s")

for i in range(n_polos_complexos):
    wn_p = params_p_comp_opt[2 * i]
    zp = params_p_comp_opt[2 * i + 1]
    print(f"Par de Polos Complexos {i+1}: wn = {wn_p:.2f} rad/s ({wn_p/(2*np.pi):.2f} Hz)  |  zeta = {zp:.3f}")

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
    'dt': sys_tf.dt if sys_tf.dt is not None else 0,
    'estrutura': {
        'n_zeros_reais': n_zeros_reais,
        'n_zeros_complexos': n_zeros_complexos,
        'n_polos_reais': n_polos_reais,
        'n_polos_complexos': n_polos_complexos,
        'tem_zero_origem': tem_zero_origem,
        'tem_polo_origem': tem_polo_origem,
    },
}

diretorio, nome_base = os.path.split(BODE_NAME_FILE)
os.makedirs(diretorio if diretorio else '.', exist_ok=True)
nome_arquivo = os.path.join(diretorio, f"{PARTE_DO_SISTEMA}_TF_wn_{nome_base}.json")

with open(nome_arquivo, 'w') as f:
    json.dump(data_export, f, indent=4)

print(f'Função de transferência exportada para {nome_arquivo}')

# =============================================================================
# 6. PLOTANDO O BODE COMPARATIVO
# =============================================================================

plt.figure()
ct.bode_plot([sys_frd, sys_tf, sys_zpk], omega=omega, dB=True, Hz=True, legend_loc='lower left')
plt.show()