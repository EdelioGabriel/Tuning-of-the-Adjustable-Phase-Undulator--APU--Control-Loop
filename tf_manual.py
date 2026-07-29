"""
IDENTIFICAÇÃO MANUAL POR EXTRAÇÃO SEQUENCIAL DE FATORES (método Nise)

Diferente dos scripts de ajuste automático (least_squares / Optuna), aqui a ideia
é: você propõe um fator (polo, zero, par complexo, etc.), o script subtrai a
contribuição dele dos dados MEDIDOS, e mostra o resíduo (o que ainda falta
explicar). Você repete isso, fator por fator, até o resíduo virar ruído plano.

Esse método trabalha direto em cima dos dados brutos do Bode -- não depende de
nenhuma função de transferência já ajustada por outro script.

Autor: Edélio Gabriel Magalhães de Jesus
Desenvolvido com auxílio de Inteligência Artificial
"""

import control as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
import json
from functools import reduce
import operator

# =========================================================
# 1. Criação do objeto FRD a partir de um arquivo CSV
# =========================================================

BODE_NAME_FILE = './bode_files_PAPU/Id_1_Pos_without_oversampling_kp_10_Tn_0_20.csv'
PARTE_DO_SISTEMA = 'Process'   # 'Process', 'Open-Loop' ou 'Close-Loop'

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

mag_linear = 10 ** (gain / 20)
phase_rad = np.deg2rad(phase)
omega_dados = 2 * np.pi * frequency

sys_frd = ct.frd(
    mag_linear * np.exp(1j * phase_rad),
    omega_dados,
    name=f'{PARTE_DO_SISTEMA} Bode Data',
)

# =========================================================
# 2. Dados medidos que serão o ALVO da extração sequencial
# =========================================================

# Magnitude: 'gain' já vem em dB direto do CSV, então usamos como está
mag_db_dados = gain

# Fase: desembrulhada (unwrap) para ficar contínua -- a quebra brusca que
# aparece no CSV é artefato de display do TwinCAT, não um evento físico
fase_deg_dados = np.rad2deg(np.unwrap(phase_rad))

def achar_vale(omega, mag_db, f_baixo, f_alto):
    """Acha a frequência do mínimo local de magnitude entre f_baixo e f_alto."""
    mask = (omega >= f_baixo) & (omega <= f_alto)
    idx_local = np.argmin(mag_db[mask])
    omega_sub = omega[mask]
    mag_sub = mag_db[mask]
    return omega_sub[idx_local], mag_sub[idx_local]

f_vale1, mag_vale1 = achar_vale(omega_dados, mag_db_dados, 638, 1122)
f_vale2, mag_vale2 = achar_vale(omega_dados, mag_db_dados, 1122, 1547)

print(f"Vale entre 638 e 1122: f = {f_vale1:.1f} rad/s, mag = {mag_vale1:.1f} dB")
print(f"Vale entre 1122 e 1547: f = {f_vale2:.1f} rad/s, mag = {mag_vale2:.1f} dB")

# =========================================================
# 3. Fábrica de fatores candidatos (cada um retorna H(jw) complexo)
# =========================================================
def fator(omega, tipo, **kwargs):
    s = 1j * omega
    if tipo == 'ganho':
        H = kwargs['K'] * np.ones_like(s)
    elif tipo == 'polo_real':      # 1/(tau*s + 1), quebra em 1/tau rad/s
        tau = kwargs['tau']
        H = 1.0 / (tau * s + 1.0)
    elif tipo == 'zero_real':      # (tau*s + 1)
        tau = kwargs['tau']
        H = (tau * s + 1.0)
    elif tipo == 'polo_quad':      # wn^2 / (s^2 + 2*zeta*wn*s + wn^2)
        wn, zeta = kwargs['wn'], kwargs['zeta']
        H = wn**2 / (s**2 + 2*zeta*wn*s + wn**2)
    elif tipo == 'zero_quad':      # (s^2 + 2*zeta*wn*s + wn^2) / wn^2
        wn, zeta = kwargs['wn'], kwargs['zeta']
        H = (s**2 + 2*zeta*wn*s + wn**2) / wn**2
    elif tipo == 'polo_origem':    # 1/s
        H = 1.0 / s
    elif tipo == 'zero_origem':    # s
        H = s
    else:
        raise ValueError(f'Tipo desconhecido: {tipo}')
    return H

# =========================================================
# 4. Subtração: modelo acumulado até agora vs. dados medidos
# =========================================================
def residuo(omega, mag_db_orig, fase_deg_orig, fatores):
    """
    fatores: lista de dicts, ex:
        [{'tipo': 'polo_quad', 'wn': 1000, 'zeta': 0.95}, ...]
    Retorna o resíduo (o que ainda falta explicar) em dB e graus,
    comparado contra os dados medidos (mag_db_orig, fase_deg_orig).
    """
    H_total = np.ones_like(omega, dtype=complex)
    for f in fatores:
        tipo = f['tipo']
        params = {k: v for k, v in f.items() if k != 'tipo'}
        H_total *= fator(omega, tipo, **params)

    mag_db_modelo = 20 * np.log10(np.abs(H_total))
    fase_deg_modelo = np.rad2deg(np.unwrap(np.angle(H_total)))

    mag_residual = mag_db_orig - mag_db_modelo
    fase_residual = fase_deg_orig - fase_deg_modelo
    return mag_residual, fase_residual, mag_db_modelo, fase_deg_modelo

# =========================================================
# 5. Registro do histórico + plot único no final (grid de subplots)
# =========================================================

# Cada etapa registrada aqui vira uma linha no grid final: [mag | fase]
historico_residuos = []

def registrar_etapa(omega, mag_db_orig, fase_deg_orig, fatores, titulo=''):
    """Calcula o resíduo da etapa atual e guarda no histórico, sem plotar
    nada ainda. Retorna (mag_res, fase_res) caso você queira inspecionar
    os números direto no console entre uma etapa e outra."""
    mag_res, fase_res, _, _ = residuo(omega, mag_db_orig, fase_deg_orig, fatores)
    historico_residuos.append({
        'titulo': titulo,
        'omega': omega,
        'mag_res': mag_res,
        'fase_res': fase_res,
    })
    return mag_res, fase_res


def plot_evolucao_residuos(historico):
    """Plota todas as etapas registradas em um único grid: uma linha por
    etapa, coluna esquerda = magnitude, coluna direita = fase."""
    n = len(historico)
    if n == 0:
        print("Nenhuma etapa registrada em 'historico_residuos'.")
        return

    fig, axes = plt.subplots(n, 2, sharex=True, figsize=(11, 3 * n))

    # Garante que 'axes' seja sempre 2D, mesmo com uma única etapa
    if n == 1:
        axes = axes.reshape(1, 2)

    for i, etapa in enumerate(historico):
        ax_mag, ax_fase = axes[i, 0], axes[i, 1]

        ax_mag.semilogx(etapa['omega'], etapa['mag_res'])
        ax_mag.set_ylabel('Magnitude (dB)')
        ax_mag.grid(True, which='both')
        ax_mag.set_title(etapa['titulo'], fontsize=9)

        ax_fase.semilogx(etapa['omega'], etapa['fase_res'])
        ax_fase.set_ylabel('Fase (graus)')
        ax_fase.grid(True, which='both')

        if i == n - 1:
            ax_mag.set_xlabel('Frequência (rad/s)')
            ax_fase.set_xlabel('Frequência (rad/s)')

    plt.tight_layout()
    plt.show()

# =========================================================
# 6. Zoom nos dados BRUTOS (sem subtrair nada) ao redor de cada ressonância
# =========================================================
# Objetivo: olhar visualmente se existe um vale de magnitude (antirressonância)
# logo antes ou depois de cada pico -- sinal de um zero_quad companheiro.

def plot_zoom_ressonancias(omega, mag_db, fase_deg, freqs, meia_largura_oitavas=1.5):
    """Plota magnitude e fase BRUTAS (nenhum fator subtraído), com zoom em
    torno de cada frequência de `freqs`. Uma linha por frequência, coluna
    esquerda = magnitude, direita = fase. Uma linha vertical tracejada marca
    a frequência central de cada janela."""
    n = len(freqs)
    fig, axes = plt.subplots(n, 2, figsize=(11, 3 * n))

    if n == 1:
        axes = axes.reshape(1, 2)

    for i, f in enumerate(freqs):
        fmin = f / (2 ** meia_largura_oitavas)
        fmax = f * (2 ** meia_largura_oitavas)
        mask = (omega >= fmin) & (omega <= fmax)

        ax_mag, ax_fase = axes[i, 0], axes[i, 1]

        ax_mag.semilogx(omega[mask], mag_db[mask], marker='o', markersize=3)
        ax_mag.axvline(f, color='red', linestyle='--', linewidth=1)
        ax_mag.set_ylabel('Magnitude (dB)')
        ax_mag.set_title(f'Zoom bruto em torno de {f} rad/s ({fmin:.0f}–{fmax:.0f} rad/s)', fontsize=9)
        ax_mag.grid(True, which='both')

        ax_fase.semilogx(omega[mask], fase_deg[mask], marker='o', markersize=3)
        ax_fase.axvline(f, color='red', linestyle='--', linewidth=1)
        ax_fase.set_ylabel('Fase (graus)')
        ax_fase.grid(True, which='both')

        if i == n - 1:
            ax_mag.set_xlabel('Frequência (rad/s)')
            ax_fase.set_xlabel('Frequência (rad/s)')

    plt.tight_layout()
    plt.show()


# Checagem rápida: confirma se a janela de 1547 rad/s está sendo cortada
# pelo fim real dos dados medidos (suspeita da mensagem anterior)
print(f"omega_dados vai de {omega_dados.min():.1f} a {omega_dados.max():.1f} rad/s")

plot_zoom_ressonancias(
    omega_dados, mag_db_dados, fase_deg_dados,
    freqs=[638, 1122, 1547],
    meia_largura_oitavas=1.5,
)

# =========================================================
# 7. Uso — replicando o processo sequencial do Nise, direto nos dados
# =========================================================

fatores_extraidos = [
    {'tipo': 'polo_origem'},
    {'tipo': 'zero_real', 'tau': 1 / 55},
    {'tipo': 'polo_real', 'tau': 1 / 12.2},
    {'tipo': 'polo_quad', 'wn': 685, 'zeta': 0.02},
    {'tipo': 'zero_quad', 'wn': 883.6, 'zeta': 0.3},
    {'tipo': 'zero_quad', 'wn': 880, 'zeta': 0.7},
    {'tipo': 'polo_quad', 'wn': 1167, 'zeta': 0.02},
    {'tipo': 'zero_quad', 'wn': 1325.4, 'zeta': 0.3},
    {'tipo': 'ganho', 'K': 8.22},
]

mag_res, fase_res = registrar_etapa(omega_dados, mag_db_dados, fase_deg_dados, fatores_extraidos,
                titulo='Resíduo após extrair TUDO')

# Plota tudo de uma vez, ao final, com a evolução completa
plot_evolucao_residuos(historico_residuos)

# =========================================================
# 8. Construção da FT final a partir dos fatores extraídos
# =========================================================

def fator_para_tf(f):
    """Converte um dict de fator (mesmo formato usado em `fator()`) em um
    objeto TransferFunction do pacote `control`. Mantém a correspondência
    1:1 com o que `fator()` calcula em jw, só que agora como polinômio em s."""
    tipo = f['tipo']
    if tipo == 'ganho':
        K = f['K']
        return ct.tf([K], [1])
    elif tipo == 'polo_real':
        tau = f['tau']
        return ct.tf([1], [tau, 1])
    elif tipo == 'zero_real':
        tau = f['tau']
        return ct.tf([tau, 1], [1])
    elif tipo == 'polo_quad':
        wn, zeta = f['wn'], f['zeta']
        return ct.tf([wn**2], [1, 2 * zeta * wn, wn**2])
    elif tipo == 'zero_quad':
        wn, zeta = f['wn'], f['zeta']
        return ct.tf([1, 2 * zeta * wn, wn**2], [wn**2])
    elif tipo == 'polo_origem':
        return ct.tf([1], [1, 0])
    elif tipo == 'zero_origem':
        return ct.tf([1, 0], [1])
    else:
        raise ValueError(f'Tipo desconhecido: {tipo}')


def construir_tf(fatores):
    """Multiplica sequencialmente cada fator (control.TransferFunction) para
    montar a FT completa -- o pacote `control` já cuida da expansão/soma
    dos polinômios de numerador e denominador."""
    tfs = [fator_para_tf(f) for f in fatores]
    return reduce(operator.mul, tfs)


sys_tf = construir_tf(fatores_extraidos)
print("\nFunção de transferência identificada:")
print(sys_tf)

# =========================================================
# 9. Bode comparando os dados medidos (FRD) com a FT identificada
# =========================================================

ct.bode_plot(
    [sys_frd, sys_tf],
    omega=omega_dados,
    dB=True,
    Hz=False,
    deg=True,
    label=['Dados medidos (FRD)', 'FT identificada'],
)
plt.suptitle(f'Comparação Bode -- {PARTE_DO_SISTEMA}')
plt.show()

# =========================================================
# 10. Exportação dos coeficientes em JSON (formato num/den/dt)
# =========================================================

num_coefs = np.asarray(sys_tf.num[0][0]).tolist()
den_coefs = np.asarray(sys_tf.den[0][0]).tolist()

resultado = {
    'num': num_coefs,
    'den': den_coefs,
    'dt': 0,
}

CAMINHO_JSON = f'./tf_{PARTE_DO_SISTEMA.lower()}.json'
with open(CAMINHO_JSON, 'w', encoding='utf-8') as f:
    json.dump(resultado, f, indent=4, ensure_ascii=False)

print(f"\nCoeficientes exportados para: {CAMINHO_JSON}")