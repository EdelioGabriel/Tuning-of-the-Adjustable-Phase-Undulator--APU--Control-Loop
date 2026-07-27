# Tuning of the Adjustable Phase Undulator (APU) Control Loop

Conjunto de scripts em Python para **identificação de sistemas** a partir de diagramas de Bode, usado na sintonia da malha de controle de um Adjustable Phase Undulator (APU/PAPU) — obtidoso via TwinCAT (Beckhoff). O objetivo central é: partir de dados experimentais de resposta em frequência, ajustar uma função de transferência (FT) que os represente, recuperar os parâmetros físicos do sistema mecânico e ganhos ótimos para a malha de controle.

Todos os scripts têm como autor **Edélio Gabriel Magalhães de Jesus**.

## Fluxo de trabalho

O repositório implementa, na prática, um pipeline sequencial:

1. **Separação dos experimentos** — `bodes_separation.py`
   Lê um único `.csv` exportado diretamente do TwinCAT (contendo vários blocos de experimentos concatenados, cada um identificado por uma linha `Name`) e separa cada bloco em um arquivo `.csv` individual, salvo em `bode_files_PAPU/`.

2. **Ajuste da função de transferência a um Bode individual** — duas abordagens equivalentes:
   - `optuna_transfer_function_analysis.py`: cria um objeto FRD (Frequency Response Data) via `python-control` e usa **Optuna** para buscar automaticamente a melhor estrutura do modelo (quantidade de polos/zeros reais e complexos, presença ou não de polo/zero na origem), penalizando a complexidade via critério **AICc**. Os parâmetros contínuos de cada estrutura candidata são ajustados com `scipy.optimize.least_squares`.
   - `system_transfer_function.py`: mesma ideia, mas com a estrutura (número de polos/zeros) fixada manualmente pelo usuário em vez de buscada pelo Optuna.
   - Ambos exportam a FT resultante (numerador/denominador) para um `.json` em `tfs_json_PAPU/`.

3. **Análise manual auxiliar** — `tf_manual.py`
   Ferramenta de apoio inspirada no método sequencial de extração de fatores (livro do Nise): permite subtrair manualmente candidatos a polos/zeros (reais ou quadráticos) da curva de Bode e visualizar o resíduo restante, comparando com a FT já ajustada e salva em `tfs_json_PAPU/`.

4. **Avaliação estatística do modelo** — `aic.py`
   Recarrega os dados brutos de um Bode e a FT já ajustada (JSON) e calcula o **AICc** (Critério de Informação de Akaike corrigido para amostras pequenas), permitindo comparar objetivamente modelos com diferentes números de parâmetros.

5. **Identificação física de duas massas** — `two_mass_identification.py`
   A partir da FT ajustada, extrai os polos/zeros complexos (frequências naturais e amortecimentos) e resolve o sistema físico correspondente a um modelo motor+carga com acoplamento elástico, obtendo:
   - `J1` (inércia do motor) e `J2` (inércia da carga)
   - `K_theta` (rigidez torcional do acoplamento)
   - `D_theta` (amortecimento torcional)

   O script também compara o resultado com um valor de datasheet (quando fornecido) e sinaliza a presença de um possível ganho `Kt` implícito.

6. **Visualização comparativa** — `visualization_bodes.py`
   Carrega todos os `.csv` de `bode_files_PAPU/`, monta um objeto FRD para cada experimento, calcula margens de ganho e de fase (`ct.margin`) e plota todos os diagramas de Bode sobrepostos para comparação visual entre diferentes configurações de controle (ex.: diferentes ganhos `kp` e tempos integrais `Tn`).

## Estrutura de pastas

| Pasta/arquivo | Conteúdo |
|---|---|
| `files/` | Arquivo(s) brutos exportados diretamente do TwinCAT, antes da separação (ex.: `BodeProject_PAPU_1.csv`), com múltiplos experimentos concatenados |
| `bode_files_PAPU/` | Diagramas de Bode individuais já separados (um `.csv` por experimento/configuração de controle) |
| `tfs_json_PAPU/` | Funções de transferência ajustadas, exportadas em `.json` (`num`, `den`, e metadados de estrutura quando aplicável) |
| `bodes_separation.py` | Separa o CSV bruto do TwinCAT em arquivos individuais por experimento |
| `optuna_transfer_function_analysis.py` | Ajuste de FT com busca automática de estrutura (Optuna + AICc) |
| `system_transfer_function.py` | Ajuste de FT com estrutura de polos/zeros fixada manualmente |
| `tf_manual.py` | Extração manual sequencial de fatores (polos/zeros) por análise de resíduo |
| `aic.py` | Cálculo do AICc de uma FT ajustada frente aos dados originais |
| `two_mass_identification.py` | Identificação dos parâmetros físicos (J1, J2, K_theta, D_theta) do modelo de duas massas |
| `visualization_bodes.py` | Plot comparativo dos Bodes de todos os experimentos, com margens de ganho e fase |

## Dependências principais

`python-control`, `numpy`, `pandas`, `scipy`, `matplotlib`, `optuna`.
