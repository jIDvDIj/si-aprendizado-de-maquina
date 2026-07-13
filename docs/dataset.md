# O Dataset — Stroke Prediction Dataset

> Este documento descreve **os dados em si**: origem, estrutura, qualidade e
> particularidades — e, separadamente, **o que foi alterado neles e por quê**.
> Para o código que implementa cada alteração, ver `docs/fase1-preprocessamento.md`;
> para a metodologia completa, `docs/relatorio.md`.

## 1. Origem e formato

- **Fonte:** *Stroke Prediction Dataset*, publicado no Kaggle.
- **Arquivo:** `data/healthcare-dataset-stroke-data.csv`.
- **Tamanho bruto:** **5.110 registros × 12 colunas**, um paciente por linha.
- **Unidade de análise:** um paciente e seus dados clínicos/demográficos no
  momento do registro, mais um rótulo indicando se ele teve AVC.
- O dataset **não documenta** a origem populacional (país, período de coleta,
  critério de amostragem) — uma limitação que se propaga para qualquer
  conclusão clínica tirada do modelo (ver `docs/relatorio.md`, Seção 7).

## 2. As 12 colunas, uma a uma

| Coluna | Tipo no arquivo | Descrição | Valores observados |
|---|---|---|---|
| `id` | inteiro | Identificador único do paciente | Sem repetições; sem relação com o alvo |
| `gender` | texto | Gênero | Female: 2.994 · Male: 2.115 · **Other: 1** |
| `age` | decimal | Idade em anos | mín. 0,08 · mediana 45,0 · máx. 82,0 |
| `hypertension` | inteiro (0/1) | Tem diagnóstico de hipertensão | 0: 4.612 · 1: 498 |
| `heart_disease` | inteiro (0/1) | Tem alguma doença cardíaca | 0: 4.834 · 1: 276 |
| `ever_married` | texto | Já foi casado(a) | Yes: 3.353 · No: 1.757 |
| `work_type` | texto | Tipo de ocupação | Private: 2.925 · Self-employed: 819 · children: 687 · Govt_job: 657 · Never_worked: 22 |
| `Residence_type` | texto | Tipo de residência | Urban: 2.596 · Rural: 2.514 |
| `avg_glucose_level` | decimal | Glicose média no sangue (mg/dL) | mín. 55,12 · mediana 91,89 · máx. 271,74 |
| `bmi` | decimal | Índice de Massa Corporal | mín. 10,30 · mediana 28,10 · máx. 97,60 · **201 ausentes** |
| `smoking_status` | texto | Status de tabagismo | never smoked: 1.892 · Unknown: 1.544 · formerly smoked: 885 · smokes: 789 |
| `stroke` | inteiro (0/1) | **Variável-alvo**: teve AVC | 0: 4.861 (95,13%) · 1: 249 (4,87%) |

Todos os números acima foram **medidos diretamente no arquivo carregado**
(não assumidos de antemão) — o script imprime essas contagens antes de
qualquer tratamento, exatamente para que nenhuma decisão de limpeza se baseie
em suposição.

## 3. O que os dados mostram antes de qualquer tratamento

- **Alvo fortemente desbalanceado:** só 4,87% dos pacientes tiveram AVC. Essa
  é a característica que mais influencia as decisões metodológicas do
  projeto — da escolha do SMOTE à escolha do Recall como métrica de decisão.
- **`bmi` tem 201 valores ausentes** (3,9% do total) — a única coluna com
  nulos; todas as outras 11 estão 100% preenchidas.
- **Zero linhas duplicadas** (linha inteira idêntica a outra).
- **Um único registro com `gender = "Other"`** — as demais 5.109 linhas são
  Female ou Male.
- **Os fatores de risco clínicos já aparecem no dado bruto, antes de qualquer
  modelo:** pacientes com hipertensão têm taxa de AVC de **13,25%** contra
  **3,97%** dos sem hipertensão (~3,3× maior); pacientes com doença cardíaca,
  **17,03%** contra **4,18%** (~4,1× maior). Esses números servem de
  referência para verificar, mais adiante, se um modelo treinado está
  captando a direção correta desse sinal.
- **Escalas muito diferentes entre as numéricas:** `avg_glucose_level` chega
  a 271,74 enquanto `age` não passa de 82 — sem normalização, um algoritmo
  baseado em distância (KNN) daria peso desproporcional à glicose.
- **`smoking_status = "Unknown"` é a segunda categoria mais frequente**
  (1.544 registros, ~30% do total) — não é um dado ausente tecnicamente (é um
  valor de categoria válido no dataset), mas representa ausência de
  informação sobre o hábito de fumar de quase um terço dos pacientes.

## 4. O que foi alterado — e por quê

O tratamento se divide em duas famílias de operação, com naturezas e
momentos de execução diferentes (ver `docs/fase1-preprocessamento.md`,
Seção 1.6, para o porquê dessa separação ser estrutural no código):

### 4.1 Limpeza estrutural (aplicada uma vez, sobre o dataset inteiro)

| Alteração | Números | Justificativa |
|---|---|---|
| **Remoção da coluna `id`** | 12 → 11 colunas | Identificador sem relação com o alvo; mantê-la arriscaria o modelo "decorar" pacientes específicos em vez de aprender um padrão. |
| **Remoção de duplicatas** | 0 linhas removidas (nenhuma encontrada) | Verificado, não assumido — o dataset já chegou sem duplicatas. |
| **Remoção do registro `gender = "Other"`** | 5.110 → 5.109 linhas (perda de 0,02%) | Uma única instância não permite a nenhum modelo aprender um padrão para essa categoria, e sua presença quebraria a codificação binária planejada para `gender` (que assume exatamente 2 categorias). Manter como terceira categoria one-hot adicionaria uma coluna quase sempre zero, sem ganho de informação. |

Depois dessas três operações, o dataset de trabalho tem **5.109 linhas e 11
colunas** (10 variáveis preditoras + o alvo `stroke`).

### 4.2 Transformação numérica (aprendida só no treino, dentro do pipeline)

Diferente da limpeza acima, estas operações **calculam alguma estatística a
partir dos dados** — por isso só podem ser ajustadas com o conjunto de
treino, depois do split, para não vazar informação do teste:

| Alteração | Colunas afetadas | Justificativa |
|---|---|---|
| **Imputação dos 201 nulos de `bmi` pela mediana** | `bmi` | A mediana é robusta a *outliers*: o IMC tem uma cauda longa (máximo 97,6 contra mediana 28,1), então a média seria puxada para cima por poucos valores extremos. Remover as 201 linhas foi descartado por eliminar ~4% dos dados — e o teste abaixo mostra que essa perda é muito mais grave do que parece à primeira vista. |
| **One-Hot Encoding** em `work_type` (5 categorias) e `smoking_status` (4 categorias) | 2 colunas de texto → 9 colunas binárias | São categorias **sem ordem natural**. Numerá-las (Label Encoding) criaria uma hierarquia falsa — ex.: `Private = 3 > Govt_job = 1` — que o KNN (distância) e a Árvore (limiares) interpretariam como diferença de magnitude real, quando não existe nenhuma. |
| **Codificação binária 0/1** em `gender`, `ever_married`, `Residence_type` | 3 colunas de texto → 3 colunas binárias | Com exatamente 2 categorias cada, uma única coluna 0/1 já captura toda a informação (equivalente a Label Encoding, mas sem risco de ordem falsa, já que só há duas opções). |
| **Normalização MinMax (0–1)** em `age`, `avg_glucose_level`, `bmi` | as 3 numéricas contínuas | Corrige a diferença de escala apontada na Seção 3 — essencial para o KNN, que soma diferenças ao quadrado (ou absolutas) entre colunas para calcular distância: sem escala comum, a coluna de maior magnitude (glicose) dominaria o resultado. |
| **`hypertension` e `heart_disease` passam direto** | nenhuma mudança | Já nascem como 0/1 no arquivo original — não há texto para converter nem escala para ajustar. |

Resultado: cada paciente passa a ser representado por **17 variáveis
numéricas**, todas em escala comparável — o formato que KNN e Árvore de
Decisão de fato recebem (ver a lista completa das 17 colunas em
`docs/fase1-preprocessamento.md`, Seção 1.5).

### 4.2.1 Teste: o que aconteceria se as 201 linhas fossem removidas em vez de imputadas

A justificativa acima ("descartaria ~4% dos dados, incluindo casos raros de
AVC") foi verificada empiricamente, e não apenas argumentada:

**Achado 1 — a ausência de `bmi` não é aleatória em relação ao alvo.** Dos
201 pacientes com `bmi` nulo, **40 tiveram AVC** — uma taxa de **19,9%**
dentro desse subgrupo, quase **4× a taxa geral do dataset (4,87%)**. Isso já
é um problema sério por si só: os 40 casos representam **16,1% de todos os
249 casos de AVC do dataset inteiro**. A ausência de `bmi` está
correlacionada com o próprio alvo — plausivelmente porque pacientes mais
graves (maior risco de AVC) têm mais chance de não ter tido o IMC registrado
no atendimento.

**Achado 2 — o efeito no modelo é grande, não cosmético.** Retreinando os
dois modelos campeões com as 201 linhas **removidas** em vez de imputadas
(mesmo split proporcional, mesmo `random_state`, mesma configuração de
hiperparâmetros vencedora):

| Modelo | Cenário | Treino | Teste | Recall (AVC) |
|---|---|---:|---:|---:|
| Árvore (`max_depth=5`) | Imputação (produção) | 4.087 (199 AVC) | 1.022 (50 AVC) | **0,8200** |
| Árvore (`max_depth=5`) | Remoção das 201 linhas | 3.926 (167 AVC) | 982 (42 AVC) | **0,5476** |
| KNN (K=15, Euclidiana, uniforme) | Imputação (produção) | 4.087 (199 AVC) | 1.022 (50 AVC) | **0,5200** |
| KNN (K=15, Euclidiana, uniforme) | Remoção das 201 linhas | 3.926 (167 AVC) | 982 (42 AVC) | **0,4762** |

Remover as linhas em vez de imputar **derrubaria o Recall da Árvore de 82%
para 55%** — uma queda de 27 pontos percentuais, muito mais grave do que a
perda proporcional de linhas (~4%) sugeriria. O KNN também piora, embora
menos (52% → 48%). A causa é exatamente o Achado 1: a remoção não é uma
amostragem aleatória do dataset — ela desproporcionalmente remove os
pacientes de maior risco, exatamente a classe que o projeto mais precisa
detectar. Isso confirma, com números e não apenas argumento, que a imputação
pela mediana foi a decisão correta.

### 4.3 Balanceamento da classe rara (só no treino, depois do split)

| Alteração | Números | Justificativa |
|---|---|---|
| **SMOTE** (oversampling sintético da classe `stroke = 1`) | Treino: 199 → 3.888 exemplos positivos (igualando os 3.888 negativos) | Sem essa correção, um classificador aprende que prever sempre "sem AVC" já minimiza o erro — o próprio dataset comprova isso: um modelo trivial nesse padrão atinge 95,1% de acurácia com 0% de Recall (Seção 5.1 de `docs/relatorio.md`). O SMOTE gera exemplos sintéticos interpolando entre casos reais de AVC, para que o modelo veja exemplos suficientes da classe rara durante o treino. |

Esta é uma alteração **de natureza diferente** das anteriores: ela não muda o
formato de nenhuma coluna (isso já foi resolvido na Seção 4.2) — muda a
**quantidade de exemplos** de uma classe específica, e só no recorte de
treino. O conjunto de teste nunca é alterado, para que as métricas reportadas
reflitam a proporção real de ~4,9% do mundo real (ver `docs/fase2-modelagem.md`,
Seção 2.2, para a distinção completa entre esta etapa e o encoding da
Seção 4.2).

## 5. O que não foi alterado (decisão consciente)

- **`smoking_status = "Unknown"` foi mantida como categoria própria**, não
  tratada como nulo nem imputada. É um valor legítimo do dataset (o paciente
  pode genuinamente não ter informado o hábito), e tratá-la como ausente
  exigiria uma estratégia de imputação sem base — por isso ela simplesmente
  vira mais uma coluna no One-Hot Encoding de `smoking_status`.
- **Nenhum paciente foi removido por ser "outlier"** em `age`, `avg_glucose_level`
  ou `bmi` (ex.: BMI = 97,6). Em um contexto clínico, valores extremos podem
  ser exatamente os casos de maior risco — removê-los enviesaria o modelo
  contra os pacientes mais graves.
