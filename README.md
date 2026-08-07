# Tuning of the Adjustable Phase Undulator (APU) Control Loop

Conjunto de scripts em Python para **identificação de sistemas** a partir de diagramas de Bode, usado na sintonia da malha de controle de um Adjustable Phase Undulator (APU) — dados obtidos via TwinCAT (Beckhoff). O objetivo central é: partir de dados experimentais de resposta em frequência, ajustar uma função de transferência (FT) que os represente e estimar os ganhos ótimos para a malha de controle, em complementação à função de *autotuning* da fabricante, permitindo maior interpretabilidade do processo.

---

**Projeto desenvolvido durante as férias de verão de 2026, no Departamento Adjunto de Tecnologia do Centro Nacional de Pesquisa em Energia e Materiais (CNPEM), sob orientação de Rafael Batista Cardoso, Gerente do grupo de Automação e Robótica (ARO).**

---

## Visão geral

O pipeline parte de um Bode experimental (magnitude + fase em função da frequência) e caminha em duas frentes complementares:

1. **Identificação matemática** — ajustar uma função de transferência (polos/zeros) que reproduza a curva medida, por diferentes métodos (mínimos quadrados, busca de estrutura com Optuna, extração manual de fatores, Vector Fitting).
2. **Identificação dos ganhos ótimos** — análise dos diagramas de Bode em malha aberta para inferir a estabilidade do sistema em malha fechada e, a partir daí, obter os ganhos que melhor sintonizem a malha do ondulador.

A identificação física do sistema por um modelo aproximado de duas massas (`two_mass_identification.py`) foi a abordagem inicial do projeto, mas está marcada como **obsoleta**: o modelo não se ajustou adequadamente aos dados reais. Ainda assim, o script é mantido no repositório por seu valor de referência teórica e documentação do processo.

---

## Estrutura do projeto

```
.
├── bode_files_PAPU/              # Bodes individuais já separados (um .csv por experimento)
├── bode_adjust_results/          # Resultados das funções de transferência ajustadas aos dados pelos métodos empregados
├── files/                        # Arquivos .csv brutos exportados do TwinCAT (múltiplos experimentos concatenados)
├── models_config_tfs_json/       # Arquivos .json com as configurações de polos e zeros usadas pelos métodos manuais
├── scope_view_files/             # Arquivos .csv brutos exportados do TwinCAT (experimentos de resposta a um step, em função do tempo)
├── scope_view_results/           # Arquivos .png com os dados da resposta ao step e estatísticas calculadas
├── tfs_json_PAPU/                # Arquivos .json com os coeficientes das funções de transferência ajustadas
├── tunning_results/              # Resultados do processo de tuning
│
├── bode_separator.py             # Separa o arquivo bruto dos experimentos de Bode em arquivos individuais
├── bode_visualizer.py            # Visualização inicial e comparativa dos experimentos já separados
├── tf_identifier.py              # Ajuste da função de transferência via Mínimos Quadrados Não-Lineares (scipy.optimize.least_squares)
├── optuna_tf_identifier.py       # Ajuste da função de transferência via otimização de estrutura com Optuna (seleção por AICc)
├── tf_manual.py                  # Ajuste manual da função de transferência, baseado no método de extração sequencial de fatores
├── vector_fitting_identifier.py  # Ajuste da função de transferência via Vector Fitting
├── methods_comparator.py         # Comparação dos erros RMS entre os ajustes obtidos pelos diferentes métodos
├── two_mass_identification.py    # Identificação dos parâmetros físicos do modelo aproximado de 2 massas (OBSOLETO)
├── time_domain_simulation.py     # Simulação da resposta no domínio do tempo a partir da FT ajustada
├── scope_visualizer.py           # Plots dos dados de resposta ao step (scope)
├── tunning_analyzer.py           # Análise dos resultados do processo de tuning
├── plot_style.py                 # Padroniza as configurações de plotagem
```

---

## Instalação

**Pré-requisitos:** Python 3.10+

```bash
git clone https://github.com/EdelioGabriel/Tuning-of-the-Adjustable-Phase-Undulator--APU--Control-Loop.git
cd Tuning-of-the-Adjustable-Phase-Undulator--APU--Control-Loop
pip install -r requirements.txt
```

---

## Fluxo completo

### 1. Separação dos experimentos de Bode

Lê um único `.csv` exportado diretamente do TwinCAT — contendo vários blocos de experimentos concatenados, cada um identificado por uma linha `Name` — e separa cada bloco em um arquivo individual:

```bash
python bode_separator.py
```

Saída: um `.csv` por experimento em `bode_files_PAPU/`.

### 2. Visualização inicial

```bash
python bode_visualizer.py
```

Carrega os experimentos já separados e gera uma visualização inicial das curvas de Bode, antes de qualquer ajuste de modelo.

### 3. Ajuste da função de transferência

Quatro abordagens disponíveis para ajustar cada Bode individual — escolha a mais adequada ao caso:

```bash
python tf_identifier.py              # estrutura de polos/zeros fixada manualmente + least_squares
python optuna_tf_identifier.py       # busca automática de estrutura (Optuna + AICc)
python tf_manual.py                  # extração sequencial de fatores (polos/zeros retirados manualmente da curva)
python vector_fitting_identifier.py  # ajuste via Vector Fitting
```

Todos criam um objeto FRD (Frequency Response Data) via `python-control`, ajustam os parâmetros da FT candidata aos dados medidos e exportam o resultado (numerador/denominador) para `tfs_json_PAPU/` (e, quando aplicável, a configuração de polos/zeros usada em `models_config_tfs_json/`).

### 4. Comparação entre métodos

```bash
python methods_comparator.py
```

Compara o erro RMS entre os ajustes obtidos pelos diferentes métodos do passo anterior, apoiando a escolha do modelo mais adequado para cada Bode.

### 5. Identificação física de duas massas *(obsoleto)*

```bash
python two_mass_identification.py
```

A partir da FT ajustada, extrai os polos/zeros complexos (frequências naturais e amortecimentos) e tenta resolver o sistema físico motor + carga com acoplamento elástico:

| Parâmetro | Significado |
|---|---|
| `J1` | Inércia do motor |
| `J2` | Inércia da carga |
| `K_θ` | Rigidez torcional do acoplamento |
| `D_θ` | Amortecimento torcional |

Mantido como referência teórica — os parâmetros físicos estimados por este modelo não reproduziram adequadamente a resposta em frequência medida.

### 6. Simulação no domínio do tempo

```bash
python time_domain_simulation.py
```

Simula a resposta da FT ajustada no domínio do tempo, permitindo comparar com os dados de resposta ao step coletados em `scope_view_files/`.

### 7. Visualização dos dados de step (scope)

```bash
python scope_visualizer.py
```

Gera os plots da resposta ao step a partir dos arquivos brutos em `scope_view_files/`, salvando os resultados com estatísticas em `scope_view_results/`.

### 8. Visualização comparativa dos Bodes e margens de estabilidade

```bash
python bode_visualizer.py
```

Carrega todos os `.csv` de `bode_files_PAPU/`, monta um objeto FRD para cada experimento, calcula margens de ganho e de fase (`ct.margin`) e plota todos os diagramas de Bode sobrepostos — útil para comparar diferentes configurações de controle (ex.: ganhos `kp` e tempos integrais `Tn`).

### 9. Análise do tuning

```bash
python tunning_analyzer.py
```

Analisa os resultados do processo de tuning (ganhos estimados, margens de estabilidade, etc.), consolidando-os em `tunning_results/`.

---

## Autor

Edélio Gabriel Magalhães de Jesus

Alguns scripts foram desenvolvidos com auxílio de ferramentas de Inteligência Artificial.
