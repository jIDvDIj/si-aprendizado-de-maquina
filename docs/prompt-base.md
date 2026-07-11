# Prompt: Desenvolvimento do Projeto "Sistema Inteligente de Classificação — Previsão de AVC"

> Copie tudo o que está abaixo da linha e cole para o Claude. O prompt é autocontido: contém o contexto, todas as decisões técnicas já tomadas e os entregáveis esperados.

---

## PAPEL E OBJETIVO

Você é um engenheiro de machine learning e vai desenvolver, **de ponta a ponta**, um projeto acadêmico de classificação para um curso de Análise e Desenvolvimento de Sistemas (ADS). Todas as decisões de projeto já foram tomadas e estão especificadas abaixo — **siga-as fielmente**, não as reabra. Seu trabalho é implementar tudo com código limpo, organizado e funcional, e produzir o relatório crítico.

O tema é a **previsão de risco de AVC (Acidente Vascular Cerebral)** a partir de dados clínicos e demográficos de pacientes, tratando um problema de **classificação binária desbalanceada**.

## CONTEXTO E CRITÉRIOS DE AVALIAÇÃO

O projeto é avaliado em três eixos, e seu trabalho deve maximizar cada um:

- **Qualidade do código (30%)**: organização, boas práticas, e — crítico — **separação de conceitos**: o código de treino fica totalmente separado do código da interface.
- **Análise científica / métricas (40%)**: rigor na avaliação. A matriz de confusão e as métricas devem ser corretamente calculadas, interpretadas e discutidas. Este é o eixo de maior peso.
- **Relatório e apresentação (30%)**: objetividade, clareza e linguagem técnica adequada.

## DATASET

- **Fonte**: Stroke Prediction Dataset (Kaggle), arquivo `healthcare-dataset-stroke-data.csv` (~5.110 registros).
- **Ambiente-alvo**: Google Colab. Assuma que o CSV será carregado via upload manual (`files.upload()`) ou pela API do Kaggle; deixe o carregamento parametrizável por um caminho de arquivo.
- **Colunas**:
  - `id` — identificador único (remover; sem valor preditivo).
  - `gender` — Male / Female / Other (categórica binária no tratamento; ver encoding).
  - `age` — numérica contínua.
  - `hypertension` — 0/1 (já numérica; manter).
  - `heart_disease` — 0/1 (já numérica; manter).
  - `ever_married` — Yes / No (categórica binária).
  - `work_type` — Private / Self-employed / Govt_job / children / Never_worked (nominal).
  - `Residence_type` — Urban / Rural (categórica binária).
  - `avg_glucose_level` — numérica contínua.
  - `bmi` — numérica contínua, **com valores ausentes** (aparecem como "N/A" no arquivo; trate a conversão).
  - `smoking_status` — formerly smoked / never smoked / smokes / Unknown (nominal).
  - `stroke` — **variável-alvo** binária (0 = sem AVC, 1 = com AVC). Classe positiva é rara (~5%).
- **Atenção**: verifique o número real de nulos em `bmi` após carregar (não assuma um valor fixo). Há também 1 registro com `gender = "Other"` — decida um tratamento simples e documente.

## DECISÕES TÉCNICAS OBRIGATÓRIAS

### Algoritmos
Usar **dois** algoritmos: **KNN (K-Nearest Neighbors)** e **Árvore de Decisão**. Não usar Naive Bayes.

### Fase 1 — Pré-processamento
- Remover a coluna `id`.
- Remover registros duplicados.
- **Nulos de `bmi`**: imputar pela **mediana** (robusta a outliers; não remover linhas).
- **Encoding**:
  - **One-Hot Encoding** nas nominais sem ordem: `work_type`, `smoking_status`.
  - **Label / binário** nas de duas categorias: `gender`, `ever_married`, `Residence_type`.
  - `hypertension` e `heart_disease` já são 0/1 — manter como estão.
- **Escala**: **MinMaxScaler** (normalização 0–1) nas numéricas `age`, `avg_glucose_level`, `bmi`. (Crítico para o KNN, que é baseado em distância.)
- **Split**: treino/teste **80/20**, com `stratify` na variável-alvo e `random_state` fixo (para reprodutibilidade).
- **Desbalanceamento**: aplicar **SMOTE apenas no conjunto de treino**, depois do split, usando o **`Pipeline` da biblioteca `imbalanced-learn`** (não o `Pipeline` do scikit-learn). O pipeline aplica o SMOTE só no `fit` e o ignora na predição, evitando vazamento de dados. Requer `pip install imbalanced-learn`.

### Fase 2 — Modelagem
Explorar hiperparâmetros por **laço manual** (não usar GridSearchCV — a escolha é didática, para facilitar a explicação).
- **KNN**: testar `K = 3, 5, 7` combinado com as distâncias **Euclidiana e Manhattan** → 6 combinações.
- **Árvore de Decisão**: testar `max_depth = 3, 5, 10 e None (sem limite)` → 4 configurações, demonstrando o efeito da profundidade sobre o overfitting.
- Como o balanceamento vem do SMOTE, **não** usar `class_weight` na árvore.

### Fase 3 — Avaliação
- Para **cada** modelo: gerar **matriz de confusão** e calcular **acurácia, precisão, recall e F1-Score**.
- **Métrica decisória para eleger o melhor modelo**: **Recall da classe AVC** (classe positiva). Justificativa: em contexto clínico, o falso negativo — não detectar quem terá AVC — é o erro mais grave.
- **Acurácia** deve ser reportada, mas explicitamente discutida como **enganosa em dados desbalanceados** (mostrar que um modelo trivial que prevê "sem AVC" para todos teria ~95% de acurácia e seria inútil).
- **Apresentação dos resultados**: **heatmaps** das matrizes de confusão + uma **tabela comparativa** consolidando as métricas de todos os modelos lado a lado.

### Fase Bônus — Deployment
- Interface web em **Streamlit**, onde o usuário insere os dados de um paciente e recebe a predição em tempo real.
- Modelo salvo com **`joblib`**, como **pipeline completo** (pré-processamento + modelo em um único arquivo `.joblib`), para que a interface receba dados brutos e o pipeline aplique a mesma transformação do treino.

## ENTREGÁVEIS

Produza os seguintes artefatos, com **separação clara entre treino e interface**:

1. **Notebook / script de treino** (ex.: `treino_modelo.ipynb` ou `treino_modelo.py`), contendo, em seções bem comentadas:
   - Carregamento e análise exploratória breve (distribuição do alvo, nulos, tipos).
   - Todo o pré-processamento da Fase 1.
   - O split e o SMOTE via pipeline.
   - O treino e avaliação de todas as combinações da Fase 2 (laço manual).
   - Toda a avaliação da Fase 3 (matrizes de confusão como heatmaps + tabela comparativa de métricas).
   - A seleção do melhor modelo pela métrica decisória e o salvamento do pipeline final em `.joblib`.

2. **Aplicação Streamlit** (ex.: `app.py`), que:
   - Carrega o pipeline `.joblib` (sem nenhuma lógica de treino).
   - Apresenta um formulário com os campos do paciente (idade, glicose, IMC, gênero, tipo de trabalho, status de fumante, hipertensão, doença cardíaca, etc.).
   - Retorna a predição (com AVC / sem AVC) e, se possível, a probabilidade associada.

3. **Relatório crítico** (texto, pronto para colar em um documento Word), contendo:
   - Breve descrição do problema, do dataset e da metodologia (pode se basear nas decisões acima).
   - As tabelas e a interpretação dos resultados.
   - A justificativa fundamentada de **qual modelo foi o melhor e por quê**, ancorada no Recall e na discussão do desbalanceamento.
   - Discussão do efeito do SMOTE e da diferença de desempenho entre KNN e Árvore.

4. **`requirements.txt`** com as dependências (pandas, numpy, scikit-learn, imbalanced-learn, matplotlib, seaborn, streamlit, joblib).

## REQUISITOS DE QUALIDADE (mapeados aos critérios)

- **Código (30%)**: funções bem nomeadas e reutilizáveis; comentários explicando o *porquê* das decisões; nenhuma lógica de treino dentro do `app.py`; código que roda sem erros no Colab.
- **Métricas (40%)**: cálculos corretos; interpretação de cada métrica no contexto clínico; discussão explícita de por que a acurácia engana aqui; comparação rigorosa entre os modelos.
- **Relatório (30%)**: linguagem objetiva e técnica; tabelas e gráficos legíveis; conclusão clara e defensável.

## FORMATO DE SAÍDA ESPERADO

Entregue os arquivos completos e prontos para uso, na ordem: (1) script/notebook de treino, (2) `app.py`, (3) relatório crítico, (4) `requirements.txt`. Antes do código, faça um resumo de 3–4 linhas do plano. Ao longo do código, use comentários que sirvam de apoio para eu explicar o trabalho na apresentação. Se precisar assumir algo não especificado, escolha a opção mais simples e didática e deixe a suposição comentada.