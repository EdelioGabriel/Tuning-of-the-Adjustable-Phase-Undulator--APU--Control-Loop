import control as ct
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.optimize as opt
import io
import json

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

# =========================================================
# Extração da FT a partir do arquivo .json
# =========================================================

NOME_ARQUIVO = "./tfs_json_PAPU/TF_Id_1_Vel_NC_kp_586_Tn_15__1.csv.json"

try:
    with open(NOME_ARQUIVO, 'r') as arquivo:
        dados = json.load(arquivo)

        sys_tf = ct.tf(dados['num'], dados['den'])
        print(f"Modelo lido do arquivo '{NOME_ARQUIVO}':")
        print(sys_tf)

except FileNotFoundError:
    raise FileNotFoundError(f"Arquivo '{NOME_ARQUIVO}' não encontrado.")

# Vetor de frequências original (dos dados medidos)
omega_dados = sys_frd.omega

# Vetor estendido para os modelos analíticos (ex: 1 década a mais em cada ponta)
omega_min_ext = omega_dados.min() / 100
omega_max_ext = omega_dados.max() * 10
omega_ext = np.logspace(np.log10(omega_min_ext), np.log10(omega_max_ext), 500)

# Resposta em frequência dos modelos analíticos no range estendido
resp_tf = ct.frequency_response(sys_tf, omega_ext)
mag_tf = 20 * np.log10(np.abs(resp_tf.fresp[0, 0]))
fase_tf = np.rad2deg(np.unwrap(np.angle(resp_tf.fresp[0, 0])))

# =========================================================
# 2. Fábrica de fatores candidatos (cada um retorna H(jw) complexo)
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
# 3. Subtração: modelo acumulado até agora vs. dados originais
# =========================================================
def residuo(omega, mag_db_orig, fase_deg_orig, fatores):
    """
    fatores: lista de dicts, ex:
        [{'tipo': 'polo_quad', 'wn': 1000, 'zeta': 0.95}, ...]
    Retorna o resíduo (o que ainda falta explicar) em dB e graus.
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
# 4. Plot no estilo do livro (dados menos modelo já extraído)
# =========================================================
def plot_residuo(omega, mag_residual, fase_residual, titulo=''):
    fig, (ax_mag, ax_fase) = plt.subplots(2, 1, sharex=True, figsize=(7, 6))

    ax_mag.semilogx(omega, mag_residual)
    ax_mag.set_ylabel('Magnitude (dB)')
    ax_mag.grid(True, which='both')
    ax_mag.set_title(titulo)

    ax_fase.semilogx(omega, fase_residual)
    ax_fase.set_ylabel('Fase (graus)')
    ax_fase.set_xlabel('Frequência (rad/s)')
    ax_fase.grid(True, which='both')

    plt.tight_layout()
    plt.show()

# =========================================================
# 5. Uso — replicando o processo sequencial do Nise
# =========================================================

# Etapa 1: seu fator quadrático já proposto
fatores_extraidos = [
    {'tipo': 'polo_origem'},
]

mag_res, fase_res, _, _ = residuo(omega_ext, mag_tf, fase_tf, fatores_extraidos)
plot_residuo(omega_ext, mag_res, fase_res, titulo='Resíduo após extrair o polo na origem')

fatores_extraidos.append({'tipo': 'polo_quad', 'wn': 100, 'zeta': 0.25})

mag_res, fase_res, _, _ = residuo(omega_ext, mag_tf, fase_tf, fatores_extraidos)
plot_residuo(omega_ext, mag_res, fase_res, titulo='Resíduo após extrair o polo complexo')

'''
fatores_extraidos.append({'tipo': 'polo_origem'})

mag_res, fase_res, _, _ = residuo(omega_ext, mag_tf, fase_tf, fatores_extraidos)
plot_residuo(omega_ext, mag_res, fase_res, titulo='Resíduo após extrair o polo complexo')
'''