# Tuning of the Adjustable Phase Undulator (APU) Control Loop

Conjunto de scripts em Python para **identificação de sistemas** a partir de diagramas de Bode, usado na sintonia da malha de controle de um Adjustable Phase Undulator (APU/PAPU) — obtidos via TwinCAT (Beckhoff). O objetivo central é: partir de dados experimentais de resposta em frequência, ajustar uma função de transferência (FT) que os represente e estimar os ganhos ótimos para a malha de controle, em complementação à função de *autotuning* da fabricante, permitindo maior interpretabilidade do processo.

´´´
**Projeto desenvolvido durante as férias de verão de 2026, no Departamento Adjunto de Tecnologia do Centro Nacional de Pesquisa em Energia e Materiais (CNPEM), sob orientação de Rafael Batista Cardoso, Gerente do grupo de Automação e Robótica (ARO)**
´´´´

---

## Visão geral

O pipeline parte de um Bode experimental (magnitude + fase em função da frequência) e caminha em duas frentes complementares:

1. **Identificação matemática** — ajustar uma função de transferência (polos/zeros) que reproduza a curva medida.
3. **Identificação dos ganhos ótimos** - análise dos diagramas de Bode para inferi a estabilidade do sistema em malha fechada a partir dos dados de malha aberta para, então, obter os ganhos que melhor sintonizem a malha do ondulador.
---

## Estrutura do projeto

```
.
├── bode_files_PAPU/                         # Pasta dos bodes individuais já separados (um .csv por experimento)
├── bode_adjust_results/                     # Pasta dos resultados das funções de transferência ajustadas aos dados pelos métodos empregados
├── files/                                   # Pastas dos arquivos .csv brutos exportados do TwinCAT (múltiplos experimentos concatenados)
├── models_config_tfs_json/                  # Pasta dos arquivos .json com as configurações de polos e zeros usadas pelos métodos manuais
├── scope_view_files/                        # Pasta dos arquivos.csv brutos exportados do TwinCAT (experimentos de resposta a um step, em função do tempo)
├── scope_view_results/                      # Pasta dos arquivos .png com os dados da resposta em frequência com estatísticas calculadas
├── tfs_json_PAPU/                           # Pasta dos arquivos .json com os coeficientes das funções de transferência ajustadas
├── tunning_results/                         # Pasta dos resultados do processo de tuning
├── bode_separator.py                        # Script para separar o arquivo bruto dos experimentos de Bode
├── bode_visualizer.py                       # Script para visualização inicial dos experimentos de separados
├── methods_comparator.py                    # Script para comparar os erros RMS entre os ajustes obtidos
├── optuna_tf_identifier.py                  # Script para realizar o ajuste da função de transferência via otimização com Optuna
├── scope_visualizer.py                      # Script para gerar os plots dos dados da resposta so step
├── tf_identifier.py                         # Script para ajuste da função de tranfereência via Mínimos Quadrados Não-Lineares (do scipy)
├── tf_manual.py                             # Script para ajuste manual da função de tranferência, baseado no método de extração de fatores mínimos
├── time_domain_simulation.py                # Script para simulação da resposta no domínio do tempo
├── tunning_analyzer                         # Script para analizar os resultados do tuning
├── two_mass_identification.py               # Script para realizar a identificação dos parâmetros físicos d modelo aproximado (OBSOLETO)
├── vector_fitting_identifier.py             # Script para ajustar a função de transferência via Vector Fitting
```
## Instalação

**Pré-requisitos:** Python 3.10+

```bash
git clone https://github.com/EdelioGabriel/Tuning-of-the-Adjustable-Phase-Undulator--APU--Control-Loop.git
pip install -r requirements
```

---

## Fluxo completo

### 1. Separação dos experimentos

Lê um único `.csv` exportado diretamente do TwinCAT — contendo vários blocos de experimentos concatenados, cada um identificado por uma linha `Name` — e separa cada bloco em um arquivo individual:

```
python bodes_separation.py
```

Saída: um `.csv` por experimento em `bode_files_PAPU/`.

### 2. Ajuste da função de transferência

Duas abordagens equivalentes, escolha uma para cada Bode individual:

```
python optuna_transfer_function_analysis.py   # busca automática de estrutura (Optuna + AICc)
python system_transfer_function.py            # estrutura de polos/zeros fixada manualmente
```

Ambos criam um objeto FRD (Frequency Response Data) via `python-control`, ajustam os parâmetros contínuos com `scipy.optimize.least_squares` e exportam o resultado (numerador/denominador) para `tfs_json_PAPU/`.

### 3. Análise manual auxiliar

```
python tf_manual.py
```

Ferramenta de apoio inspirada no método sequencial de extração de fatores (livro do Nise): subtrai manualmente candidatos a polos/zeros (reais ou quadráticos) da curva de Bode e mostra o resíduo restante, comparando com a FT já ajustada.

### 4. Avaliação estatística do modelo

```
python aic.py
```

Recarrega os dados brutos de um Bode e a FT já ajustada (JSON), calculando o **AICc** (Critério de Informação de Akaike corrigido para amostras pequenas) — permite comparar objetivamente modelos com diferentes números de parâmetros.

### 5. Identificação física de duas massas

```
python two_mass_identification.py
```

A partir da FT ajustada, extrai os polos/zeros complexos (frequências naturais e amortecimentos) e resolve o sistema físico motor + carga com acoplamento elástico:

| Parâmetro | Significado |
|---|---|
| `J1` | Inércia do motor |
| `J2` | Inércia da carga |
| `K_θ` | Rigidez torcional do acoplamento |
| `D_θ` | Amortecimento torcional |

### 6. Visualização comparativa

```
python visualization_bodes.py
```

Carrega todos os `.csv` de `bode_files_PAPU/`, monta um objeto FRD para cada experimento, calcula margens de ganho e de fase (`ct.margin`) e plota todos os diagramas de Bode sobrepostos — útil para comparar diferentes configurações de controle (ex.: ganhos `kp` e tempos integrais `Tn`).

---
