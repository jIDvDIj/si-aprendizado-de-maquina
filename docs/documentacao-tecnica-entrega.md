# Documentação Técnica de Entrega — Previsão de AVC

> Escopo deste documento: registrar, de forma técnica e honesta, **todas as
> decisões de projeto e seus porquês**, **como KNN e Árvore de Decisão foram
> efetivamente aplicados**, **o que funcionou, o que falhou, o que foi testado
> e descartado**, e **o que pode/não pode ser feito daqui para frente**. Os
> documentos `docs/fase1-preprocessamento.md` a `docs/fase-bonus-deployment.md`
> explicam o código linha a linha; este documento explica a **trajetória de
> engenharia** por trás dele — incluindo os becos sem saída.

## 1. Objetivo e critério de sucesso

Prever risco de AVC a partir de dados clínicos/demográficos (Stroke
Prediction Dataset, Kaggle), com **dois algoritmos obrigatórios** (KNN e
Árvore de Decisão) e **classe positiva rara** (4,9% dos 5.109 pacientes após
limpeza). O critério de sucesso definido para o projeto **não é acurácia**: é
**Recall da classe AVC** — a proporção de pacientes que de fato terão AVC e
que o modelo consegue identificar — porque, no domínio clínico, o falso
negativo (dizer "sem risco" para quem terá AVC) é o erro mais caro possível.
Essa escolha de métrica está por trás de quase toda decisão registrada abaixo.

## 2. Decisões de arquitetura e o porquê de cada uma

| Decisão | Alternativa descartada | Por quê |
|---|---|---|
| Imputação de `bmi` por **mediana** | Remover linhas nulas / imputar por média | Mediana é robusta à cauda longa do IMC (máx. 97,6 vs. mediana 28,1); remover linhas descartaria ~4% dos dados, incluindo AVCs raros |
| **One-Hot Encoding** em `work_type`/`smoking_status` | Label Encoding | Label Encoding criaria ordem numérica falsa (Private=3 > Govt_job=1) que KNN e Árvore interpretariam como magnitude real |
| **MinMaxScaler** (não `StandardScaler`) | Padronização Z-score | Deixa todas as variáveis no mesmo intervalo [0,1] fechado, o que também simplifica a leitura dos limiares de decisão da árvore e mantém o KNN livre de outliers extremos dominando a escala |
| `ColumnTransformer` ajustado **dentro do pipeline** | Transformar o dataset inteiro antes do split | Evita vazamento de dados: mediana, min/máx e categorias são aprendidos só no treino |
| `Pipeline` do **imbalanced-learn** (não scikit-learn) | `Pipeline` padrão + SMOTE manual antes do split | É o único jeito de garantir, por construção, que o SMOTE nunca roda em `predict` nem contamina o teste |
| Laço manual de hiperparâmetros | `GridSearchCV` | Exigência didática do enunciado — cada combinação deve ser uma chamada explícita e visível |
| **Recall** como métrica decisória (não acurácia/F1) | Otimizar acurácia | O baseline trivial ("nunca prever AVC") atinge 95,1% de acurácia com Recall 0 — provado no próprio código, não hipotético (Seção 5 abaixo) |
| **Sem `class_weight`** na Árvore | Combinar `class_weight="balanced"` com SMOTE | As duas técnicas de balanceamento se sobreporiam, distorcendo a comparação entre configurações |
| **Dois modelos** entregues na interface (não só o vencedor) | Embarcar só a Árvore | Permite comparar na prática o comportamento conservador da Árvore contra o mais parcimonioso do KNN; a Árvore continua marcada como "recomendado" |

### 2.1 Duas linhas da tabela acima que costumam ser confundidas

O One-Hot Encoding e o SMOTE aparecem na mesma tabela, mas **resolvem
problemas independentes**, em pontos diferentes do pipeline — vale explicitar
o porquê, já que a confusão entre os dois é comum:

| | Encoding (`OneHotEncoder`, binário 0/1) | SMOTE |
|---|---|---|
| **Problema que resolve** | KNN e Árvore só operam sobre números; texto (`"Private"`, `"Female"`) não entra em uma conta de distância nem em um limiar de corte. | Apenas 4,9% dos pacientes de treino têm `stroke = 1`; sem correção, o modelo aprende que "prever sempre sem AVC" já minimiza o erro. |
| **O que muda** | O **formato** de cada valor (texto → número). Não altera quantas linhas de cada classe existem. | A **quantidade de exemplos** da classe minoritária (via exemplos sintéticos). Não altera o significado de nenhuma coluna. |
| **Onde age** | Todas as colunas de entrada (`X`), em qualquer paciente. | Só o conjunto de treino, só depois do split, e nunca a coluna-alvo isolada. |
| **Ordem de execução** | Primeiro (Fase 1) — é pré-requisito matemático do SMOTE. | Depois (Fase 2) — o SMOTE interpola distâncias entre vizinhos, o que só é definido sobre números; não há "meio-termo" matemático entre dois textos. |

Fazer só o encoding (sem SMOTE) deixaria o desbalanceamento intocado — os
modelos ainda aprenderiam majoritariamente a classe 0. Fazer o SMOTE sem o
encoding não é sequer possível de implementar: o algoritmo calcula distância
euclidiana entre vizinhos para interpolar valores, uma operação indefinida
sobre strings.

## 3. Como o KNN foi efetivamente aplicado

### 3.1 Primeira rodada (linha de base do enunciado)

Conforme o enunciado, a primeira exploração cobriu `K ∈ {3, 5, 7}` ×
`distância ∈ {Euclidiana, Manhattan}` — 6 combinações. Resultado: o Recall
cresceu **de forma monotônica** com K nas duas distâncias (26%→38% na
Euclidiana, 22%→40% na Manhattan), com o melhor caso em K=7/Manhattan (40%).

**Leitura crítica do próprio resultado:** uma curva ainda subindo em K=7 é
evidência de que o teto do KNN não tinha sido encontrado — parar ali seria
reportar um "melhor KNN" artificialmente fraco só porque o enunciado sugeriu
`{3, 5, 7}` como exemplo, não como limite. Isso motivou a segunda rodada.

### 3.2 Segunda rodada (extensão motivada por evidência)

Ampliação para `K ∈ {3, 5, 7, 9, 11, 15}` × `distância ∈ {Euclidiana,
Manhattan}` × **novo parâmetro** `weights ∈ {uniform, distance}` — 24
combinações. Duas mudanças, duas perguntas:

1. **K maior encontra o teto?** Sim: o Recall continuou subindo até K=15
   (52% na Euclidiana/uniforme — o novo melhor KNN, ante 40% da rodada
   anterior). Não foi testado K > 15 por já estar bem acima do sugerido no
   enunciado e por o ganho marginal entre K=11 e K=15 já ser pequeno
   (44%→52% Euclidiana, mas 38%→38% Manhattan uniforme — ver CSV completo).
2. **Pesar mais o vizinho mais próximo ajuda a achar a classe rara?** A
   hipótese era que `weights="distance"` favoreceria a detecção da classe
   minoritária perto da fronteira de decisão. **A hipótese não se confirmou**:
   em praticamente todas as combinações, `weights="distance"` teve Recall
   igual ou **pior** que `weights="uniform"` (ex.: K=15 Euclidiana, 52%
   uniforme vs. 44% por distância). Interpretação: mesmo após o SMOTE, o
   vizinho isoladamente mais próximo de um paciente de risco tende a ser um
   caso majoritário; dar-lhe peso extra reforça o voto errado em vez de
   corrigi-lo — ampliar quantos vizinhos "votam" generalizou melhor do que
   concentrar peso no mais próximo.

**Resultado final do KNN:** `K=15, distância Euclidiana, peso uniforme`,
Recall = 52% (26 de 50 casos de AVC detectados), muito abaixo da Árvore (82%).
Ver tabela completa das 24 combinações em `resultados/resultados_comparativos.csv`
e os heatmaps em `resultados/fig_matrizes_knn_uniforme.png` /
`fig_matrizes_knn_distancia.png`.

## 4. Como a Árvore de Decisão foi efetivamente aplicada

`max_depth ∈ {3, 5, 10, None}` — 4 configurações, sem alterações de escopo
(o enunciado já cobria bem o espaço relevante e o resultado confirmou isso).

| `max_depth` | Recall | Leitura |
|---|---:|---|
| 3 | 0,82 | Empata com 5; árvore rasa já captura o sinal dominante (idade) |
| 5 | **0,82** | Empate desfeito pelo F1 (menos falsos positivos: 314 vs 338) — **vencedora** |
| 10 | 0,64 | Início do overfitting: começa a decorar ruído do treino |
| None (sem limite) | 0,24 | Overfitting severo — maior acurácia de teste (0,8718) e pior Recall, ilustrando por que a acurácia engana |

A demonstração de overfitting por profundidade é um dos poucos experimentos
do projeto que confirmou a expectativa teórica de forma limpa, sem
surpresas — ver a discussão de interpretabilidade (importância de variáveis,
`age` = 88,5%) no `docs/fase3-avaliacao.md`.

## 5. O que funcionou (acertos)

- **A escolha do Recall como métrica decisória** se provou correta e foi
  demonstrada com números, não só argumentada: o baseline trivial bate todos
  os 28 modelos em acurácia (95,11%) e tem Recall 0 — o próprio dataset
  comprova por que acurácia sozinha enganaria aqui.
- **A separação SMOTE-só-no-treino via `Pipeline` do imbalanced-learn**
  funcionou exatamente como projetado: a ablação (Seção 3.5 do
  `fase3-avaliacao.md`) mostra que, sem SMOTE, o Recall da Árvore desaba de
  82% para 4% — prova de que o desenho evita tanto o vazamento de dados
  quanto a inutilidade do modelo.
- **A extensão do grid do KNN foi uma decisão acertada**: sem ela, o projeto
  reportaria um "melhor KNN" de 40% de Recall quando na verdade existe uma
  configuração de 52% dentro do mesmo espaço de busca razoável — subestimar
  o concorrente da Árvore enfraqueceria a comparação final.
- **A arquitetura de pipeline único (pré-processamento + SMOTE + modelo)**
  permitiu salvar e reusar os modelos em produção sem duplicar nenhuma linha
  de lógica de treino no `app.py` — validado com testes headless
  (`streamlit.testing.v1.AppTest`) para os dois modelos e dois perfis de
  paciente (alto e baixo risco), sem falhas.
- **A escolha de `max_depth=5` sobre `max_depth=3`** via desempate por F1 foi
  a decisão certa de engenharia fina: mesmo poder de detecção, 24 falsos
  positivos a menos.

## 6. O que falhou / problemas encontrados

### 6.1 Bug real: SMOTE corrompe variáveis binárias (o achado mais importante)

Durante testes manuais do `app.py`, foi observado um comportamento
clinicamente absurdo: **em alguns perfis de paciente, marcar "tem
hipertensão" ou "tem doença cardíaca" diminuía a probabilidade de AVC
prevista**, em vez de aumentá-la.

**Causa raiz identificada:** `hypertension` e `heart_disease` são colunas
binárias (0/1), mas o `SMOTE` **interpola linearmente todas as colunas**
entre dois vizinhos da classe minoritária — inclusive essas. O resultado são
pacientes sintéticos com valores como `hypertension = 0,37`, que não existem
na realidade. Medição direta: **~7% dos valores sintéticos de
`hypertension`** e **~5% de `heart_disease`** ficavam fora de `{0, 1}` no
treino balanceado.

A árvore vencedora da época aprendeu a explorar esse ruído — sua estrutura
usava `hypertension` em **três limiares diferentes**, incluindo um
logicamente impossível:

```
hypertension <= 0.01
hypertension >  0.01
    hypertension <= 1.00
    hypertension >  1.00      ← limiar impossível para um paciente real
        hypertension <= 0.50
        hypertension >  0.50
```

O corte `> 1.00` só existe por artefato numérico da interpolação, e o
conjunto desses três cortes empurrava alguns perfis de paciente para o lado
errado da árvore — a inversão observada no app.

### 6.2 Tentativa de correção: SMOTENC

O `SMOTENC` resolve exatamente esse problema: para colunas marcadas como
categóricas, substitui a interpolação linear por **voto de maioria entre os
vizinhos**, preservando valores discretos em todo exemplo sintético.
Aplicado com 14 das 17 colunas pós-transformação marcadas como categóricas
(`hypertension`, `heart_disease` e as dummies one-hot), **o bug foi
confirmado corrigido**: `np.unique` sobre `hypertension` pós-balanceamento
passou a retornar exatamente `[0, 1]`, e o corte `hypertension > 1.00`
desapareceu da árvore.

**Efeito colateral inesperado — o Recall caiu bastante:**

| Modelo | Recall com SMOTE | Recall com SMOTENC |
|---|---:|---:|
| Árvore (`max_depth=3`) | 0,82 | 0,66 |
| Árvore (`max_depth=5`, vencedora) | 0,82 | **0,58** |
| Árvore (`max_depth=10`) | 0,64 | 0,50 |
| Árvore (`max_depth=None`) | 0,24 | 0,28 |
| Melhor KNN | 0,52 | 0,56 |

**Mecanismo identificado:** com o SMOTE comum, a corrupção fracionária fazia
`hypertension` se comportar como uma variável **contínua**, dando à árvore
rasa múltiplos pontos de corte utilizáveis (0,01 / 0,50 / 1,00) na mesma
variável para reduzir a impureza Gini em vários níveis ao mesmo tempo — uma
flexibilidade que uma variável genuinamente binária nunca ofereceria (só
existe um corte possível, 0,5). Essa flexibilidade extra aparentemente
ajudava, por acaso e não por sinal clínico real, a capturar mais casos
positivos do teste. Evidência de suporte: sob SMOTE comum, aumentar a
profundidade de 3 para 5 **não piorava** o Recall (empate em 0,82); sob
SMOTENC, o Recall já cai de 3 para 5 (0,66→0,58) — sem essa "muleta" de
ruído, cada nível extra de profundidade ajusta ruído genuíno em vez de sinal.

O KNN se moveu na direção oposta (+4 p.p.) porque a mesma corrupção distorce
o cálculo de distância; com o SMOTENC preservando valores discretos, as
distâncias passaram a refletir melhor a similaridade real entre pacientes.

### 6.3 Decisão tomada e seu custo assumido

**Optou-se por manter o `SMOTE` comum**, priorizando o Recall (a métrica
decisória oficial do projeto) sobre a correção do bug de inversão:

- **Ganho ao manter o SMOTE:** Recall de 82% em vez de 66% na Árvore
  recomendada — 41 vs. 33 casos de AVC detectados em 50, uma diferença
  clinicamente relevante.
- **Custo aceito, e que deve ser disclosed na apresentação:** o bug de
  inversão em variáveis binárias **permanece latente no modelo entregue**.
  Pacientes com hipertensão ou doença cardíaca podem, em perfis específicos,
  receber probabilidade de AVC menor do que deveriam. Isso é uma limitação
  conhecida, não um erro não identificado — ver `docs/nota-smote-vs-smotenc.md`
  para a investigação completa.

### 6.4 Problema de ambiente: segfault pandas 3 + Streamlit

Separado do bug de modelagem: o `pandas 3.0` (novo backend de string via
PyArrow) causa uma falha nativa (segmentation fault) na thread de execução
do Streamlit neste ambiente — tanto ao montar o `DataFrame` de uma linha
quanto em `st.dataframe`/`st.table`. Não é um problema de lógica, é
incompatibilidade binária entre bibliotecas nesta configuração de SO/WSL2.
**Resolvido** fixando `pandas>=2.0,<3.0` no `requirements.txt` (mesma série
usada no Google Colab) e trocando `st.dataframe` por `st.json` na exibição
de dados de transparência do app — ver `docs/fase-bonus-deployment.md`,
Seção 4.7.

## 7. Possibilidades testadas (resumo objetivo)

| Testado | Resultado | Destino |
|---|---|---|
| KNN K∈{3,5,7} × 2 distâncias (6 combos) | Recall subindo monotonicamente, teto não alcançado | Superado pela rodada seguinte |
| KNN K∈{3,5,7,9,11,15} × 2 distâncias × 2 pesos (24 combos) | Melhor Recall = 52% (K=15, Euclidiana, uniforme) | **Mantido** — resultado final do KNN |
| `weights="distance"` no KNN | Recall igual ou pior que `weights="uniform"` na maioria das combinações | **Descartado** — hipótese refutada pelos dados |
| Árvore `max_depth∈{3,5,10,None}` | Overfitting claro a partir de 10; 3 e 5 empatam em Recall | **Mantido** — 5 vencedor por desempate de F1 |
| `class_weight="balanced"` combinado com SMOTE | Não testado formalmente — descartado por desenho (sobreposição de balanceamento) | **Descartado por decisão de projeto** |
| SMOTENC (correção da corrupção de variáveis binárias) | Corrigiu o bug, mas derrubou o Recall da Árvore de 82% para 58% | **Testado e revertido** — Recall priorizado |
| `monotonic_cst` no `DecisionTreeClassifier` (restringir sinal de `hypertension`/`heart_disease`) | — | **Não implementado** (ver Seção 8) |

## 8. O que pode ser feito (próximos passos concretos)

Em ordem de custo/benefício estimado:

1. **Combinar SMOTENC + `monotonic_cst`** (scikit-learn ≥ 1.4): manter a
   correção estrutural do SMOTENC (sem interpolação fracionária) e usar a
   restrição monotônica para forçar explicitamente que `hypertension` e
   `heart_disease` só possam **aumentar** a probabilidade prevista de AVC.
   Isso devolveria ao modelo, por regularização direcionada com conhecimento
   de domínio, parte do Recall perdido ao corrigir o bug — sem depender de
   ruído sintético para isso. É a linha de investigação mais promissora
   identificada e **não foi implementada nem testada**.
2. **Calibração de limiar de decisão** (hoje fixo em 0,5 via `predict()`):
   usar `predict_proba` com um limiar mais baixo aumentaria ainda mais o
   Recall às custas de mais falsos positivos — uma curva Precision-Recall
   deixaria esse trade-off explícito e ajustável por perfil de uso (triagem
   vs. diagnóstico).
3. **Validação cruzada estratificada** no lugar de um único split 80/20 —
   reduziria a variância da estimativa de Recall reportada (com apenas 50
   casos positivos no teste, cada paciente pesa 2 pontos percentuais).
4. **Algoritmos de conjunto** (Random Forest, Gradient Boosting) — não
   pedidos pelo enunciado, mas naturalmente mais robustos ao tipo de
   overfitting observado na Árvore sem limite de profundidade.
5. **Investigar K > 15 no KNN** com `weights="uniform"`, já que o Recall
   ainda não tinha sinal claro de platô em todas as combinações — o ganho
   esperado é modesto e não muda a conclusão (KNN perde para a Árvore), mas
   fecharia a pergunta em aberto da Seção 3.2.

## 9. O que não pode ser feito (restrições reais, não preguiça)

- **Não é possível eliminar o trade-off Precisão × Recall** do modelo
  vencedor sem mudar o critério de decisão: a Precisão de 11,6% é
  consequência direta de otimizar para Recall em uma classe que é 4,9% dos
  dados — não é um bug corrigível, é a natureza do problema.
- **Não é possível usar `GridSearchCV`** (ou qualquer busca automatizada de
  hiperparâmetros) sem contrariar a exigência explícita do enunciado do
  projeto por um laço manual didático — essa é uma restrição de escopo
  acadêmico, não técnica.
- **Não é possível remover o SMOTE** sem inviabilizar o problema: a ablação
  (Seção 3.5 do `fase3-avaliacao.md`) mostra Recall caindo para 2–4% sem
  balanceamento — abaixo disso, os modelos convergem para o baseline trivial.
- **Não é possível corrigir totalmente a corrupção de variáveis binárias do
  SMOTE sem sacrificar Recall** com as ferramentas testadas até agora — a
  Seção 6.3 documenta esse trade-off como decisão consciente, não como
  limitação ignorada. A mitigação da Seção 8.1 (`monotonic_cst`) é uma
  hipótese não testada, não uma solução comprovada.
- **Não é possível validar o modelo clinicamente** com os dados e o escopo
  deste projeto: o dataset não documenta a origem populacional, tem apenas
  249 casos positivos no total, e nenhuma revisão médica/regulatória foi
  feita. A ferramenta é estritamente acadêmica — isso é uma limitação de
  escopo do projeto, não algo que mais engenharia resolveria.
- **Não é possível garantir reprodutibilidade bit-a-bit fora do
  `RANDOM_STATE=42` fixado** — mudanças de versão de scikit-learn/imbalanced-learn
  entre ambientes (local vs. Colab) podem produzir pequenas variações
  numéricas nos exemplos sintéticos do SMOTE, mesmo com a semente fixa;
  o `requirements.txt` fixa faixas de versão para minimizar esse risco, mas
  não o elimina por completo.

## 10. Estado de entrega

| Artefato | Papel |
|---|---|
| `treino_modelo.py` / `notebooks/treino_modelo.ipynb` | Treino completo: 28 modelos avaliados, 2 salvos |
| `modelos/modelo_arvore.joblib` | Pipeline recomendado (Recall 82%) |
| `modelos/modelo_knn.joblib` | Pipeline alternativo (Recall 52%) |
| `app.py` | Interface Streamlit com seletor entre os dois modelos |
| `docs/relatorio.md` | Relatório crítico (síntese para entrega/apresentação) |
| `docs/fase1..4*.md` | Documentação técnica por fase (código linha a linha) |
| `docs/nota-smote-vs-smotenc.md` | Investigação completa do bug da Seção 6.1–6.3 |
| **`docs/documentacao-tecnica-entrega.md`** (este arquivo) | Síntese de decisões, testes, falhas e limitações |

**Recomendação para a apresentação:** divulgar explicitamente o bug da
Seção 6 como um achado do processo de investigação, não escondê-lo — ele
demonstra rigor de avaliação (o eixo de maior peso do projeto, 40%) melhor
do que qualquer resultado limpo, precisamente porque mostra a decisão de
priorizar Recall sendo tomada com pleno conhecimento do seu custo.
