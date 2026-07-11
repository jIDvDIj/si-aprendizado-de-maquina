# Relatório Crítico — Sistema Inteligente de Classificação: Previsão de AVC

## 1. Definição do problema

O Acidente Vascular Cerebral (AVC) é uma das principais causas de morte e incapacidade no mundo, e sua prevenção depende de identificar precocemente pacientes de risco. Este projeto constrói e compara modelos de *machine learning* capazes de prever, a partir de dados clínicos e demográficos simples (idade, hipertensão, glicose média, IMC, tabagismo etc.), se um paciente pertence ao grupo de risco de AVC.

Trata-se de um problema de **classificação binária fortemente desbalanceada**: apenas **4,9%** dos pacientes do dataset tiveram AVC. Esse desbalanceamento condiciona todas as decisões metodológicas do trabalho — do balanceamento por SMOTE à escolha da métrica de avaliação —, pois um modelo ingênuo que "chuta" sempre a classe majoritária aparenta excelente desempenho sem ter nenhuma utilidade clínica (Seção 5.1).

## 2. Dataset

Foi utilizado o *Stroke Prediction Dataset* (Kaggle), arquivo `healthcare-dataset-stroke-data.csv`, com **5.110 registros** e 12 colunas. Características relevantes identificadas na análise exploratória:

| Aspecto | Valor medido |
|---|---|
| Registros / colunas | 5.110 / 12 |
| Alvo `stroke` = 1 (com AVC) | 249 pacientes (4,9%) |
| Alvo `stroke` = 0 (sem AVC) | 4.861 pacientes (95,1%) |
| Valores ausentes em `bmi` (gravados como "N/A") | 201 (3,9% dos registros) |
| Registros duplicados | 0 |
| Registros com `gender` = "Other" | 1 |

As variáveis dividem-se em numéricas contínuas (`age`, `avg_glucose_level`, `bmi`), binárias já numéricas (`hypertension`, `heart_disease`), categóricas binárias (`gender`, `ever_married`, `Residence_type`) e categóricas nominais (`work_type`, com 5 categorias; `smoking_status`, com 4). A coluna `id` é um identificador sem valor preditivo.

## 3. Metodologia

### 3.1 Pré-processamento (Fase 1)

| Decisão | Justificativa |
|---|---|
| Remoção de `id` | Identificador único; não carrega padrão algum. |
| Remoção de duplicatas | Boa prática; no dataset havia 0 (verificado, não assumido). |
| Remoção do único registro `gender = "Other"` | Uma única instância não permite aprender um padrão para a categoria e quebraria a codificação binária de `gender`; perda de 0,02% dos dados, documentada. |
| Imputação dos 201 nulos de `bmi` pela **mediana** | A mediana é robusta a *outliers* (a média seria puxada pelos valores extremos de IMC); remover as linhas descartaria ~4% dos dados, incluindo casos raros de AVC. |
| **One-Hot Encoding** em `work_type` e `smoking_status` | São nominais sem ordem; *label encoding* criaria uma ordem artificial (ex.: Private = 3 > Govt_job = 1) que os modelos interpretariam como magnitude. |
| Codificação **binária 0/1** em `gender`, `ever_married`, `Residence_type` | Com apenas duas categorias, uma única coluna 0/1 é suficiente e equivale ao *label encoding*. |
| **MinMaxScaler** (normalização 0–1) em `age`, `avg_glucose_level`, `bmi` | Crítico para o KNN, que compara pacientes por distância: sem escala comum, a glicose (até ~272) dominaria o cálculo e idade/IMC quase não contariam. |

Um ponto central da implementação: as transformações que **aprendem parâmetros** dos dados (mediana da imputação, mínimo/máximo da escala, categorias do encoding) foram encapsuladas em um `ColumnTransformer` **dentro do pipeline**, de modo que são ajustadas exclusivamente com o conjunto de treino — eliminando vazamento de dados — e reaplicadas de forma idêntica e automática a qualquer dado novo, inclusive na interface web.

### 3.2 Divisão treino/teste e balanceamento (SMOTE)

Os dados foram divididos em **80% treino (4.087 pacientes) e 20% teste (1.022 pacientes)** com `stratify` no alvo — ambas as partições mantêm os 4,9% de AVC — e `random_state = 42` para reprodutibilidade total.

O desbalanceamento foi tratado com **SMOTE**, que sintetiza novos exemplos da classe minoritária interpolando vizinhos próximos, **aplicado somente ao conjunto de treino, após a divisão**: as 199 amostras positivas do treino foram expandidas para 3.888, igualando a classe majoritária (3.888/3.888). O conjunto de teste permaneceu intocado, com a proporção real de ~5% — condição indispensável para que as métricas reflitam o desempenho no mundo real.

A garantia de que o SMOTE nunca "contamina" o teste vem da arquitetura: usamos o `Pipeline` da biblioteca `imbalanced-learn` (e não o do scikit-learn), que executa o passo SMOTE apenas durante o `fit` e o ignora em qualquer `predict`.

### 3.3 Modelos e hiperparâmetros (Fase 2)

Os hiperparâmetros foram explorados por **laço manual** (escolha didática que torna cada combinação explícita, em vez do GridSearchCV):

- **KNN**: K ∈ {3, 5, 7} × distância ∈ {Euclidiana, Manhattan} → 6 combinações;
- **Árvore de Decisão**: `max_depth` ∈ {3, 5, 10, None} → 4 configurações, para observar o efeito da profundidade sobre o *overfitting*.

Como o balanceamento já é responsabilidade do SMOTE, **não** se utilizou `class_weight` na Árvore — combinar as duas técnicas distorceria a comparação.

### 3.4 Métricas e critério de decisão (Fase 3)

Para cada um dos 10 modelos foram geradas a matriz de confusão e as métricas Acurácia, Precisão, Recall e F1-Score (as três últimas referidas à classe positiva, AVC). A leitura clínica de cada uma:

- **Acurácia** — proporção total de acertos; **enganosa em dados desbalanceados** (Seção 5.1).
- **Precisão (AVC)** — dos pacientes apontados como risco, quantos realmente tiveram AVC; seu complemento são os falsos positivos, cujo custo é gerar exames adicionais desnecessários.
- **Recall (AVC)** — dos pacientes que de fato tiveram AVC, quantos o modelo detectou; seu complemento são os **falsos negativos: pacientes de risco mandados para casa sem acompanhamento — o erro mais grave do cenário**.
- **F1-Score** — média harmônica entre precisão e recall, usada como desempate.

**Métrica decisória: Recall da classe AVC.** Em triagem clínica, deixar de identificar um futuro caso de AVC (falso negativo) tem custo potencialmente fatal, enquanto um alarme falso custa apenas investigação adicional. O melhor modelo é, portanto, o que maximiza o Recall, usando o F1 como critério de desempate.

## 4. Resultados

### 4.1 Tabela comparativa (conjunto de teste, ordenada pela métrica decisória)

| Modelo | Acurácia | Precisão (AVC) | Recall (AVC) | F1-Score (AVC) |
|---|---:|---:|---:|---:|
| **Árvore (max_depth=5)** ← melhor | **0,6840** | **0,1155** | **0,8200** | **0,2025** |
| Árvore (max_depth=3) | 0,6605 | 0,1082 | 0,8200 | 0,1911 |
| Árvore (max_depth=10) | 0,7407 | 0,1147 | 0,6400 | 0,1945 |
| KNN (K=7, Manhattan) | 0,8366 | 0,1274 | 0,4000 | 0,1932 |
| KNN (K=7, Euclidiana) | 0,8131 | 0,1061 | 0,3800 | 0,1659 |
| KNN (K=5, Euclidiana) | 0,8278 | 0,1062 | 0,3400 | 0,1619 |
| KNN (K=5, Manhattan) | 0,8503 | 0,1241 | 0,3400 | 0,1818 |
| KNN (K=3, Euclidiana) | 0,8552 | 0,1048 | 0,2600 | 0,1494 |
| Árvore (max_depth=None, sem limite) | 0,8718 | 0,1143 | 0,2400 | 0,1548 |
| KNN (K=3, Manhattan) | 0,8630 | 0,0982 | 0,2200 | 0,1358 |
| Baseline trivial (prevê "sem AVC" para todos) | 0,9511 | 0,0000 | 0,0000 | 0,0000 |

*(Heatmaps das 10 matrizes de confusão em `resultados/fig_matrizes_knn.png` e `resultados/fig_matrizes_arvore.png`.)*

### 4.2 Matriz de confusão do modelo vencedor

Árvore de Decisão, `max_depth=5`, sobre os 1.022 pacientes de teste:

| | Previsto: sem AVC | Previsto: AVC |
|---|---:|---:|
| **Real: sem AVC** (972) | 658 (VN) | 314 (FP) |
| **Real: AVC** (50) | **9 (FN)** | **41 (VP)** |

O modelo detectou **41 dos 50 casos reais de AVC (Recall = 82%)**, deixando escapar 9. O preço foi o excesso de cautela: 314 pacientes saudáveis sinalizados como risco (Precisão = 11,6% — de cada 100 alertas, cerca de 12 se confirmam). Em um fluxo de **triagem**, esse é um compromisso aceitável: os sinalizados seguem para avaliação médica adicional, enquanto um falso negativo significaria nenhum acompanhamento para quem viria a ter AVC.

## 5. Discussão

### 5.1 Por que a acurácia engana neste problema

O baseline trivial — "prever sem AVC para todos" — alcança **95,1% de acurácia** com **Recall zero**: não detecta um único caso da doença e é clinicamente inútil. Esse número supera a acurácia de todos os modelos treinados, o que demonstra que, com 95% dos dados em uma classe, a acurácia mede sobretudo a habilidade de acertar a classe majoritária. É exatamente o caso da Árvore vencedora: sua acurácia de 68,4% é "pior" que a do baseline, mas ela captura 82% dos casos de AVC — a inversão de leitura que justifica ancorar a decisão no Recall. A acurácia foi reportada por completude, nunca como critério.

### 5.2 Efeito do SMOTE (ablação)

Retreinamos a melhor configuração de cada algoritmo **sem** o passo SMOTE, mantendo tudo o mais idêntico:

| Modelo | Cenário | Recall (AVC) | Precisão (AVC) | Acurácia |
|---|---|---:|---:|---:|
| KNN (K=7, Manhattan) | com SMOTE | 0,40 | 0,1274 | 0,8366 |
| KNN (K=7, Manhattan) | sem SMOTE | 0,02 | 0,5000 | 0,9511 |
| Árvore (max_depth=5) | com SMOTE | 0,82 | 0,1155 | 0,6840 |
| Árvore (max_depth=5) | sem SMOTE | 0,04 | 0,4000 | 0,9501 |

Sem balanceamento, ambos os algoritmos degeneram para perto do baseline: acurácia ~95% e Recall de 2–4% (a Árvore detectaria 2 casos em 50; o KNN, 1). Com 95% dos exemplos de treino negativos, a fronteira de decisão é dominada pela classe majoritária. O SMOTE, ao equilibrar o treino (3.888/3.888), força os modelos a levarem a classe rara a sério — multiplicando o Recall por ~20 na Árvore, ao custo esperado de precisão. **O SMOTE não é um refinamento: é o que torna o problema tratável.**

### 5.3 KNN × Árvore de Decisão

A Árvore superou o KNN com folga na métrica decisória (0,82 contra 0,40). Três fatores explicam:

1. **Natureza da fronteira de decisão.** Com profundidade limitada, a árvore aprende poucas regras globais e legíveis (dominadas por idade e glicose), que generalizam bem para a classe minoritária sintetizada. O KNN decide pela vizinhança local; mesmo após o SMOTE, as regiões de teste com casos reais de AVC continuam povoadas por maioria de vizinhos saudáveis, derrubando o Recall.
2. **Espaço de alta dimensão.** Após o One-Hot, os pacientes vivem em ~17 dimensões, várias binárias — distâncias tornam-se menos discriminativas (todas as amostras ficam "quase equidistantes"), o que penaliza métodos baseados em distância.
3. **Comportamento dos hiperparâmetros.** No KNN, o Recall cresce com K (K=7 > K=5 > K=3 nas duas distâncias): vizinhanças maiores dão mais chance aos exemplos sintéticos de vencerem a votação; a distância Manhattan com K=7 foi a melhor combinação (0,40), por ser menos sensível a diferenças grandes concentradas em uma única variável. Na Árvore, o efeito da profundidade é o retrato clássico do *overfitting*: `max_depth` 3 e 5 empatam no Recall (0,82); em 10 cai para 0,64; e **sem limite despenca para 0,24** — a árvore decora ruído do treino balanceado e perde a capacidade de generalizar, ainda que sua acurácia de teste (0,8718) seja a maior entre os modelos, reforçando a Seção 5.1.

Interpretabilidade — a importância de variáveis da Árvore vencedora concentra-se em **idade (88,5%)**, seguida de gênero (4,0%), glicose média (3,3%), IMC (2,9%) e hipertensão (1,3%), coerente com os fatores de risco clínicos conhecidos e impossível de extrair de um KNN.

### 5.4 Escolha final entre as árvores empatadas

`max_depth=3` e `max_depth=5` empataram no Recall (0,82, mesmos 9 falsos negativos). O desempate pelo F1-Score favoreceu **`max_depth=5`** (0,2025 vs 0,1911), que comete 24 falsos positivos a menos com o mesmo poder de detecção — estritamente melhor no trade-off.

## 6. Modelo eleito e implantação

O pipeline vencedor — **pré-processamento + SMOTE + Árvore de Decisão (`max_depth=5`)** — foi salvo como artefato único (`modelos/modelo_avc.joblib`) via `joblib`. A interface web (Streamlit, `app.py`) apenas carrega esse artefato e lhe entrega os dados brutos do formulário: toda transformação é reaplicada internamente pelo pipeline, garantindo que a predição em produção use exatamente o processamento do treino (separação total entre treino e interface). A aplicação retorna a classe prevista e a probabilidade estimada de AVC.

## 7. Limitações e trabalhos futuros

- **Precisão baixa (11,6%)** é inerente ao ajuste pró-Recall: o modelo serve à triagem, não ao diagnóstico. Ajustar o limiar de decisão (hoje 0,5) sobre a probabilidade permitiria calibrar o compromisso Recall × Precisão conforme a capacidade do serviço de saúde.
- ~30% dos registros têm `smoking_status = "Unknown"`, tratado como categoria própria — informativo, porém ruidoso.
- O dataset (Kaggle) não documenta origem populacional; os resultados não devem ser extrapolados clinicamente. A ferramenta é acadêmica e não substitui avaliação médica.
- Extensões naturais: validação cruzada estratificada (em vez de um único split), curvas Precision-Recall para escolha de limiar, e algoritmos de conjunto (Random Forest / Gradient Boosting).

## 8. Conclusão

O projeto cumpriu as três fases propostas: pré-processamento fundamentado, exploração manual de 10 configurações de dois algoritmos e avaliação rigorosa centrada no contexto clínico. A **Árvore de Decisão com `max_depth=5`**, treinada sobre dados balanceados por SMOTE, foi eleita o melhor modelo por detectar **82% dos casos de AVC** do teste — 41 de 50 pacientes, contra 40% do melhor KNN e 0% do baseline que "vence" em acurácia. Os resultados ilustram as duas lições metodológicas centrais do trabalho: **em dados desbalanceados, a acurácia não mede utilidade** — e a métrica de avaliação deve ser escolhida pelo custo real dos erros do domínio, que aqui é o falso negativo.
