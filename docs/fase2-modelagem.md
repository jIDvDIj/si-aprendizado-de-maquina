# Fase 2 — Divisão dos Dados, Balanceamento (SMOTE) e Modelagem

> Código-fonte: `treino_modelo.py`, seções 3 e 4.

Esta fase parte do `df_limpo` produzido na Fase 1 (5.109 pacientes, ainda com
colunas brutas — a transformação numérica só acontece dentro do pipeline) e
entrega 40 modelos treinados e avaliados: 24 KNN + 16 Árvores de Decisão.

## 2.1 Divisão treino/teste

```python
X = df_limpo.drop(columns=["stroke"])
y = df_limpo["stroke"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)
```

| | Pacientes | Proporção de AVC |
|---|---:|---:|
| Treino | 4.087 (80%) | 4,9% (199 casos) |
| Teste | 1.022 (20%) | 4,9% (50 casos) |

Dois pontos de projeto aqui:

- **`stratify=y`**: sem isso, a divisão aleatória poderia sub- ou
  super-representar a classe rara (AVC) no teste, distorcendo qualquer métrica
  calculada depois. Com estratificação, treino e teste preservam os mesmos
  4,9% do dataset original.
- **`random_state=42`** fixo em todas as etapas estocásticas do projeto
  (split, SMOTE, Árvore) — qualquer pessoa que rode o script obtém exatamente
  os mesmos 4.087/1.022 pacientes e os mesmos números de todo o relatório.

## 2.2 Balanceamento com SMOTE — o porquê e o como

> **Isto não é a mesma coisa que o encoding da Fase 1.** A Fase 1 converteu
> texto em número (`gender`, `work_type` etc.) porque os algoritmos só operam
> sobre números — isso não altera quantas linhas de cada classe existem. O
> SMOTE, por sua vez, ataca um problema totalmente diferente: a **proporção**
> entre as classes do alvo. Ver a tabela comparativa em
> `docs/fase1-preprocessamento.md`, Seção 1.5, para os dois problemas lado a
> lado.

Com apenas 199 dos 4.087 pacientes de treino sendo casos de AVC (4,9%), um
classificador treinado direto tende a aprender que "prever sempre negativo"
já minimiza o erro médio — exatamente o comportamento inútil discutido na
Fase 3. O **SMOTE** (Synthetic Minority Over-sampling Technique) ataca isso
**sintetizando novos exemplos da classe minoritária**, interpolando entre
vizinhos próximos no espaço já normalizado (por isso ele entra depois do
`ColumnTransformer` no pipeline, nunca antes — o SMOTE só consegue calcular
distância/interpolação porque a Fase 1 já converteu tudo em número; texto não
tem "meio-termo" matemático).

### A regra que não pode ser violada: SMOTE só no treino, só depois do split

Se o balanceamento fosse aplicado **antes** do split (ou sobre o dataset
inteiro), exemplos sintéticos gerados a partir de pacientes que acabariam no
conjunto de teste vazariam informação para o treino — o modelo teria, na
prática, "visto" uma versão interpolada dos próprios casos de teste antes da
avaliação, inflando artificialmente as métricas. A ordem correta é:

```
dataset limpo → split 80/20 (stratify) → SMOTE (só em X_treino/y_treino) → treino do modelo
                                       ↘ X_teste/y_teste permanecem intocados, com os 4,9% reais
```

### Por que o `Pipeline` do imbalanced-learn (e não o do scikit-learn)

```python
from imblearn.pipeline import Pipeline as PipelineImblearn
from imblearn.over_sampling import SMOTE

def construir_pipeline(modelo) -> PipelineImblearn:
    return PipelineImblearn(steps=[
        ("preprocessamento", construir_preprocessador()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("modelo", modelo),
    ])
```

O `Pipeline` padrão do scikit-learn executa todo passo tanto em `fit` quanto
em `predict`/`transform`. O `Pipeline` do **imbalanced-learn** trata o SMOTE
como um passo especial de **reamostragem**: ele só age durante `fit` (ajusta
o pré-processador, gera sintéticos, treina o modelo) e é **automaticamente
ignorado** durante `predict` — a chamada `pipeline.predict(X_teste)` passa
X_teste pelo pré-processador e direto para o modelo, sem SMOTE. É essa
propriedade que garante, por construção, que o teste nunca é balanceado
artificialmente e que o mesmo objeto pode ser salvo e reusado em produção
(Fase Bônus) sem carregar lógica de treino.

### Efeito medido (bloco demonstrativo, linhas 303–318 do script)

O script materializa o "antes e depois" só para fins didáticos (o fluxo
oficial de treino usa o pipeline acima, este bloco é ilustrativo):

| | Classe 0 (sem AVC) | Classe 1 (AVC) |
|---|---:|---:|
| Treino antes do SMOTE | 3.888 | 199 |
| Treino depois do SMOTE | 3.888 | **3.888** |

O SMOTE nivela as duas classes em exatamente 3.888 exemplos cada — os 3.889
exemplos positivos "extras" são sintéticos, interpolados entre os 199 casos
reais no espaço de 17 variáveis já normalizado. O conjunto de teste continua
com 1.022 pacientes e os mesmos 50 casos reais de AVC (4,9%).

## 2.3 Laço manual de hiperparâmetros

O enunciado pede exploração por **laço manual** (não `GridSearchCV`) — decisão
didática: cada uma das 40 combinações é uma chamada explícita e visível no
código, facilitando explicar a lógica na apresentação em vez de esconder a
busca dentro de uma função de biblioteca.

### KNN — 6 valores de K × 2 métricas de distância × 2 pesos = 24 combinações

A exploração do KNN passou por uma segunda rodada, motivada pelos resultados
da primeira: com apenas `K ∈ {3, 5, 7}`, o Recall crescia **monotonicamente**
com K nas 6 combinações originais (26%→38% na Euclidiana, 22%→40% na
Manhattan) — sinal claro de que o teto ainda não tinha sido alcançado. Duas
mudanças foram então testadas juntas:

1. **Ampliar o range de K** para `{3, 5, 7, 9, 11, 15}`, para encontrar o
   ponto em que o Recall de fato para de subir (ou começa a cair).
2. **Adicionar o parâmetro `weights`** (`"uniform"` vs `"distance"`): por
   padrão, o KNN pesa todos os K vizinhos igualmente; `weights="distance"`
   pesa o voto de cada vizinho pelo inverso da distância, dando mais força ao
   vizinho mais próximo. A hipótese era que isso ajudaria a detectar a classe
   rara em regiões de fronteira onde as classes se sobrepõem mesmo após o
   SMOTE.

```python
NOME_DISTANCIA = {"euclidean": "Euclidiana", "manhattan": "Manhattan"}
NOME_PESO = {"uniform": "uniforme", "distance": "por distância"}
VALORES_K = [3, 5, 7, 9, 11, 15]
for peso in ["uniform", "distance"]:
    for metrica_distancia in ["euclidean", "manhattan"]:
        for k in VALORES_K:
            knn = KNeighborsClassifier(n_neighbors=k, metric=metrica_distancia, weights=peso)
            resultados.append(avaliar_modelo(nome, construir_pipeline(knn)))
```

Gera as 24 combinações (2 pesos × 2 distâncias × 6 valores de K). **Resultado
medido:** o Recall de fato continuou subindo até `K=15` (Euclidiana, peso
uniforme → 52%, o novo melhor KNN, contra 40% da rodada anterior), mas a
segunda hipótese não se confirmou — `weights="distance"` teve Recall **igual
ou pior** que `weights="uniform"` em praticamente todas as combinações (ex.:
K=15 Euclidiana, 52% uniforme vs 44% por distância). Interpretação: com o
SMOTE já tendo povoado a vizinhança de exemplos sintéticos da classe
minoritária, dar peso extra ao vizinho mais próximo (em geral um ponto
majoritário isolado, já que a classe positiva é rara mesmo após o
balanceamento do treino) não ajuda tanto quanto simplesmente ampliar quantos
vizinhos "votam" — daí o peso uniforme com K maior vencer. A tabela completa
com as 24 combinações está em `resultados/resultados_comparativos.csv`; os
heatmaps, em `resultados/fig_matrizes_knn_uniforme.png` (peso uniforme) e
`resultados/fig_matrizes_knn_distancia.png` (peso por distância).

### Árvore de Decisão — 4 profundidades × 4 estratégias de `max_features` = 16 combinações

```python
NOME_MAX_FEATURES = {None: "todas", "sqrt": "raiz quadrada", "log2": "log2", 0.5: "50%"}
VALORES_MAX_FEATURES = [None, "sqrt", "log2", 0.5]
for profundidade in [3, 5, 10, None]:
    for max_features in VALORES_MAX_FEATURES:
        arvore = DecisionTreeClassifier(
            max_depth=profundidade, max_features=max_features, random_state=RANDOM_STATE
        )
        resultados.append(avaliar_modelo(nome, construir_pipeline(arvore)))
```

`max_depth=None` remove o limite de profundidade — a árvore cresce até que
todas as folhas sejam puras (ou até o mínimo de amostras por folha), o cenário
clássico de **overfitting** que a Fase 3 quantifica.

O eixo `max_features` foi adicionado numa segunda rodada, motivado por uma
observação sobre a árvore vencedora original (`max_depth=5, max_features`
padrão = todas as 17 variáveis disponíveis em cada divisão): **88,5% de toda
a decisão da árvore vinha de uma única coluna**, `age`. `max_features` limita
quantas variáveis a árvore pode considerar em cada divisão individual — não
remove nenhuma coluna do dataset, só restringe o leque de candidatas a cada
nó, forçando a árvore a usar outras variáveis quando `age` não está no
subconjunto sorteado.

**Resultado medido:** a técnica funcionou — em todo o grid, a importância de
`age` caiu para entre 20% e 66% (dependendo da combinação), e a configuração
`max_depth=3, max_features="sqrt"` chegou a **superar** a árvore original em
Recall (90% vs. 82%). Mas o ganho veio com um efeito colateral só visível na
matriz de confusão completa: essa mesma configuração sinaliza **58,5% dos
pacientes saudáveis do teste como risco** (569 de 972), contra 32,3% da
árvore original — falso-alarme grande demais para um sistema de triagem.
Por isso, **a eleição do melhor modelo continua restrita à família original**
(`max_features="todas"`) no código — ver `docs/nota-max-features-idade.md`
para a investigação completa, incluindo o mecanismo por trás do trade-off e
uma alternativa intermediária não adotada.

**Decisão importante:** a Árvore **não** usa `class_weight="balanced"`. Como o
balanceamento já é responsabilidade do SMOTE dentro do próprio pipeline,
somar as duas técnicas as faria competir — o SMOTE já entrega à árvore um
treino 50/50; aplicar `class_weight` por cima re-ponderaria uma base que já
está equilibrada, distorcendo a comparação entre as configurações.

### A função que treina e mede cada combinação

```python
def avaliar_modelo(nome: str, pipeline: PipelineImblearn) -> dict:
    pipeline.fit(X_treino, y_treino)
    y_previsto = pipeline.predict(X_teste)
    return {
        "Modelo": nome,
        "Acurácia": accuracy_score(y_teste, y_previsto),
        "Precisão (AVC)": precision_score(y_teste, y_previsto, pos_label=1, zero_division=0),
        "Recall (AVC)": recall_score(y_teste, y_previsto, pos_label=1, zero_division=0),
        "F1-Score (AVC)": f1_score(y_teste, y_previsto, pos_label=1, zero_division=0),
        "matriz_confusao": confusion_matrix(y_teste, y_previsto, labels=[0, 1]),
        "pipeline": pipeline,
    }
```

Cada chamada: (1) treina o pipeline completo — pré-processamento ajustado no
treino, SMOTE aplicado no treino, modelo ajustado nos dados balanceados; (2)
prediz sobre `X_teste` (sem SMOTE, sem vazamento); (3) calcula as quatro
métricas da Fase 3, sempre com `pos_label=1` (a classe AVC é o foco clínico,
não a classe majoritária); (4) guarda o `pipeline` treinado inteiro no
dicionário de resultado — é dali que o melhor modelo é extraído e salvo, sem
precisar retreinar nada (Fase 3, Seção 7).

`zero_division=0` evita erro/aviso caso alguma combinação nunca preveja a
classe positiva (o que aconteceria, por exemplo, no baseline trivial da
Fase 3).

Ao final desta fase, a lista `resultados` contém 40 dicionários — a matéria-
prima para a tabela comparativa e os heatmaps da Fase 3.
