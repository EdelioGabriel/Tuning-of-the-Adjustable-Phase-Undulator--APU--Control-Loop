"""
 ANÁLISE DE UM ÚNICO BODE 

Esse script tem como objetivo analisar um único Bode, carregando os dados de um arquivo CSV específico e estimando a função de transferência correspondente. Ele utiliza a biblioteca `control` para criar objetos de resposta em frequência (FRD), o Optuna para buscar a melhor estrutura (número de polos/zeros reais e complexos, presença de polo/zero na origem) e o `scipy.optimize.least_squares` para ajustar os parâmetros contínuos da função de transferência dado cada conjunto de hiperparâmetros estruturais.

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
omega_data = 2 * np.pi * frequency

# Criação do objeto FRD (Frequency Response Data) usando os dados carregados
sys_frd = ct.frd(
    mag_linear * np.exp(1j * phase_rad),
    omega_data,
    name='Open-Loop Bode Data',
)

# Extração das frequências e da resposta em frequência complexa do objeto FRD
omega = sys_frd.omega  # Frequências em rad/s
fresp = np.squeeze(sys_frd.frdata)  # Resposta em frequência complexa

N_amostras = len(omega)  # Captura seus 50 pontos para a fórmula do AICc

# ==============================================================================
# 1. FUNÇÕES AUXILIARES (residuals, x0, bounds) PARAMETRIZADAS PELA ESTRUTURA
# ==============================================================================

def make_residuals(n_zeros_reais, n_zeros_complexos, n_polos_reais, n_polos_complexos,
                    tem_zero_origem, tem_polo_origem):
    """Fecha (closure) sobre omega e fresp e retorna a função de resíduos
    para a estrutura de polos/zeros especificada."""

    def residuals(params):
        K = params[0]
        idx = 1

        tau_z_real = params[idx: idx + n_zeros_reais]
        idx += n_zeros_reais

        params_z_comp = params[idx: idx + 2 * n_zeros_complexos]
        idx += 2 * n_zeros_complexos

        tau_p_real = params[idx: idx + n_polos_reais]
        idx += n_polos_reais

        params_p_comp = params[idx: idx + 2 * n_polos_complexos]

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

    return residuals


def make_x0_bounds(n_zeros_reais, n_zeros_complexos, n_polos_reais, n_polos_complexos):
    """Monta o vetor inicial x0 e os limites (bounds) para a estrutura dada."""

    x0 = [1.0]
    x0 += [0.01] * n_zeros_reais
    for _ in range(n_zeros_complexos):
        x0 += [0.01, 0.5]  # [tau_z, zeta_z]
    x0 += [0.01] * n_polos_reais
    for _ in range(n_polos_complexos):
        x0 += [0.01, 0.5]  # [tau_p, zeta_p]

    lower_bounds = [-np.inf]
    upper_bounds = [np.inf]

    for _ in range(n_zeros_reais):
        lower_bounds.append(1e-5); upper_bounds.append(10.0)
    for _ in range(n_zeros_complexos):
        lower_bounds.extend([1e-5, 0.01]); upper_bounds.extend([10.0, 1.0])
    for _ in range(n_polos_reais):
        lower_bounds.append(1e-5); upper_bounds.append(10.0)
    for _ in range(n_polos_complexos):
        lower_bounds.extend([1e-5, 0.01]); upper_bounds.extend([10.0, 2.0])

    bounds = (lower_bounds, upper_bounds)
    return x0, bounds


# ==============================================================================
# 2. BUSCA DA ESTRUTURA (Optuna) — decide quantos polos/zeros usar
# ==============================================================================

def objective(trial):
    # Definindo limites sensatos de busca para não extrapolar a ordem física
    n_polos_reais = trial.suggest_int('n_polos_reais', 0, 5)
    n_zeros_reais = trial.suggest_int('n_zeros_reais', 0, 5)
    n_polos_complexos = trial.suggest_int('n_polos_complexos', 0, 5)
    n_zeros_complexos = trial.suggest_int('n_zeros_complexos', 0, 5)
    tem_polo_origem = trial.suggest_categorical('tem_polo_origem', [True, False])
    tem_zero_origem = trial.suggest_categorical('tem_zero_origem', [True, False])

    # 1. Restrição Física: Garante que o sistema seja estritamente causal/próprio
    grau_num = n_zeros_reais + 2 * n_zeros_complexos + (1 if tem_zero_origem else 0)
    grau_den = n_polos_reais + 2 * n_polos_complexos + (1 if tem_polo_origem else 0)
    
    if grau_num > grau_den or grau_den == 0:
        raise optuna.exceptions.TrialPruned()

    residuals = make_residuals(
        n_zeros_reais, n_zeros_complexos, n_polos_reais, n_polos_complexos,
        tem_zero_origem, tem_polo_origem
    )
    x0, bounds = make_x0_bounds(n_zeros_reais, n_zeros_complexos, n_polos_reais, n_polos_complexos)

    # K representa o número total de parâmetros contínuos estimados
    K_params = len(x0)
    
    # 2. Proteção matemática: impede o Optuna de quebrar o denominador do AICc
    if N_amostras - K_params - 1 <= 0:
        raise optuna.exceptions.TrialPruned()

    try:
        res = opt.least_squares(residuals, x0, bounds=bounds, max_nfev=3000)
    except Exception:
        raise optuna.exceptions.TrialPruned()

    # Cálculo do Resíduo Quadrático (RSS) Complexo Puro
    custo_rss = np.sum(res.fun ** 2)
    
    if custo_rss <= 1e-12:
        custo_rss = 1e-12

    # 3. PENALIZAÇÃO RIGOROSA VIA AICc (Substituindo a antiga penalidade ad-hoc)
    aic = N_amostras * np.log(custo_rss / N_amostras) + 2 * K_params
    aicc = aic + (2 * K_params * (K_params + 1)) / (N_amostras - K_params - 1)
    
    return aicc


sampler = optuna.samplers.TPESampler(seed=367)
study = optuna.create_study(
    direction='minimize',
    sampler=sampler,
    study_name='polos_zeros_aicc'
)

# Aumentado para 50 trials para explorar melhor as combinações físicas com AICc
print("Buscando a estrutura ideal de polos e zeros baseada no AICc...")
study.optimize(objective, n_trials=300, show_progress_bar=True, n_jobs=-1)

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

# Fit final com tolerâncias mais apertadas que durante a busca estrutural
res = opt.least_squares(residuals, x0, bounds=bounds, max_nfev=20000, xtol=1e-12, ftol=1e-12)

K_bode = res.x[0]
idx = 1

# Zeros Reais
tau_z_opt = res.x[idx: idx + n_zeros_reais]
idx += n_zeros_reais

# Zeros Complexos [tau_z, zeta_z]
params_z_comp_opt = res.x[idx: idx + 2 * n_zeros_complexos]
idx += 2 * n_zeros_complexos

# Polos Reais
tau_p_opt = res.x[idx: idx + n_polos_reais]
idx += n_polos_reais

# Polos Complexos [tau_p, zeta_p]
params_p_comp_opt = res.x[idx: idx + 2 * n_polos_complexos]

# ==============================================================================
# 4. MONTAGEM DO MODELO ZPK (Zero-Pole-Gain)
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
prod_tau_z = (np.prod(tau_z_opt) if n_zeros_reais > 0 else 1.0) * \
             (np.prod(params_z_comp_opt[0::2] ** 2) if n_zeros_complexos > 0 else 1.0)

prod_tau_p = (np.prod(tau_p_opt) if n_polos_reais > 0 else 1.0) * \
             (np.prod(params_p_comp_opt[0::2] ** 2) if n_polos_complexos > 0 else 1.0)

K_zpk = K_bode * (prod_tau_z / prod_tau_p)

# Objeto ZPK do python-control
sys_zpk = ct.zpk(zeros_zpk, polos_zpk, K_zpk)


# ==============================================================================
# 5. MONTAGEM DA FUNÇÃO DE TRANSFERÊNCIA POLINOMIAL (Num/Den)
# ==============================================================================

# --- NUMERADOR ---
num_poly = np.array([K_bode])

for tz in tau_z_opt:
    num_poly = np.convolve(num_poly, [tz, 1.0])

for i in range(n_zeros_complexos):
    tz = params_z_comp_opt[2 * i]
    zz = params_z_comp_opt[2 * i + 1]
    num_poly = np.convolve(num_poly, [tz**2, 2 * zz * tz, 1.0])

if tem_zero_origem:
    num_poly = np.convolve(num_poly, [1.0, 0.0])  # Multiplica por 's'


# --- DENOMINADOR ---
den_poly = np.array([1.0])

for tp in tau_p_opt:
    den_poly = np.convolve(den_poly, [tp, 1.0])

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
print("\n=== ESTRUTURA ESCOLHIDA PELO OPTUNA ===")
print(f"n_zeros_reais={n_zeros_reais}, n_zeros_complexos={n_zeros_complexos}, "
      f"n_polos_reais={n_polos_reais}, n_polos_complexos={n_polos_complexos}, "
      f"tem_zero_origem={tem_zero_origem}, tem_polo_origem={tem_polo_origem}")

print("\n=== PARÂMETROS OTIMIZADOS (FIT FINAL) ===")
print(f"K (Ganho Bode) = {K_bode:.4f}")

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
nome_arquivo = os.path.join(diretorio, f"TF_{nome_base}.json")

with open(nome_arquivo, 'w') as f:
    json.dump(data_export, f, indent=4)
    
print(f'Função de transferência exportada para {nome_arquivo}')

# =============================================================================
# 6. PLOTANDO O BODE COMPARATIVO
# =============================================================================

plt.figure()
ct.bode_plot([sys_frd, sys_tf, sys_zpk], omega=omega, legend_loc='lower left')
plt.show()
