# Tuning of the Adjustable Phase Undulator (APU) Control Loop — Identificação de Sistemas via Bode

Conjunto de scripts em Python para **identificação de sistemas** a partir de diagramas de Bode, usado na sintonia da malha de controle de um Adjustable Phase Undulator (APU/PAPU) — obtidos via TwinCAT (Beckhoff). O objetivo central é: partir de dados experimentais de resposta em frequência, ajustar uma função de transferência (FT) que os represente, recuperar os parâmetros físicos do sistema mecânico e os ganhos ótimos para a malha de controle.

---

## Visão geral

O pipeline parte de um Bode experimental (magnitude + fase em função da frequência) e caminha em duas frentes complementares:

1. **Identificação matemática** — ajustar uma função de transferência (polos/zeros) que reproduza a curva medida.
2. **Identificação física** — a partir dessa FT, recuperar os parâmetros reais do sistema mecânico, modelado como duas massas (motor + carga) acopladas por um eixo com rigidez torcional.
3. **Identificação dos ganhos ótimos** - análise dos diagramas de Bode para inferi a estabilidade do sistema em malha fechada a partir dos dados de malha aberta para, então, obter os ganhos que melhor sintonizem a malha do ondulador.
---

## Estrutura do projeto

```
.
├── files/                                   # CSV bruto exportado do TwinCAT (múltiplos experimentos concatenados)
├── bode_files_PAPU/                         # Bodes individuais já separados (um .csv por experimento)
├── tfs_json_PAPU/                           # Funções de transferência ajustadas, exportadas em .json
├── bodes_separation.py                      # Separa o CSV bruto em arquivos individuais por experimento
├── optuna_transfer_function_analysis.py     # Ajuste de FT com busca automática de estrutura (Optuna + AICc)
├── system_transfer_function.py              # Ajuste de FT com estrutura de polos/zeros fixada manualmente
├── tf_manual.py                             # Extração manual sequencial de fatores (polos/zeros) por resíduo
├── aic.py                                   # Cálculo do AICc de uma FT ajustada frente aos dados originais
├── two_mass_identification.py               # Identificação física: J1, J2, K_θ, D_θ
└── visualization_bodes.py                   # Plot comparativo de todos os Bodes, com margens de ganho e fase
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

## Dependências Python

```
pip install control numpy pandas scipy matplotlib optuna
```