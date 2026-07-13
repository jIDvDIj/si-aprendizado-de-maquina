# Fase 1 — Análise Exploratória e Pré-processamento

> Código-fonte: `treino_modelo.py`, seções 1 e 2 (linhas 124–279).

Esta fase trata os dados **brutos** do Kaggle até deixá-los em um formato numérico,
completo e em escala comum — pré-requisito para qualquer um dos dois algoritmos
(KNN e Árvore de Decisão) rodar corretamente.

## 1.1 O dataset bruto

Arquivo `data/healthcare-dataset-stroke-data.csv`: **5.110 registros, 12 colunas**.

| Coluna | Tipo original | Papel |
|---|---|---|
| `id` | inteiro | identificador único — sem valor preditivo |
| `gender` | texto (Female/Male/Other) | categórica binária (na prática) |
| `age` | decimal | numérica contínua |
| `hypertension` | 0/1 | já numérica |
| `heart_disease` | 0/1 | já numérica |
| `ever_married` | texto (Yes/No) | categórica binária |
| `work_type` | texto (5 categorias) | categórica nominal |
| `Residence_type` | texto (Urban/Rural) | categórica binária |
| `avg_glucose_level` | decimal | numérica contínua |
| `bmi` | decimal, com ausentes | numérica contínua |
| `smoking_status` | texto (4 categorias) | categórica nominal |
| `stroke` | 0/1 | **variável-alvo** |

### Carregamento (código)

```python
def carregar_dados(candidatos: list[str]) -> pd.DataFrame:
    for caminho in candidatos:
        if Path(caminho).exists():
            return pd.read_csv(caminho, na_values=["N/A"])
    raise FileNotFoundError(...)
```

O parâmetro `na_values=["N/A"]` é explícito porque a coluna `bmi` traz seus
ausentes gravados como a **string literal** `"N/A"` no CSV, não como campo vazio.
O pandas já reconheceria isso por padrão, mas deixamos explícito para documentar
a origem dos nulos. `CANDIDATOS_CSV` tenta primeiro `data/...` (estrutura deste
repositório) e depois o nome puro do arquivo (caso de `files.upload()` no Colab).

## 1.2 O que a análise exploratória mediu (números reais, não assumidos)

O prompt do projeto exige medir esses valores no dado carregado, e não assumir
um número fixo — por isso o script sempre imprime o resultado antes de agir:

| Métrica | Valor medido |
|---|---|
| Distribuição do alvo | 4.861 sem AVC (95,1%) / 249 com AVC (4,9%) |
| Nulos em `bmi` | **201** registros (3,9% do total) — nenhuma outra coluna tem nulo |
| Registros duplicados (linha inteira idêntica) | **0** |
| Registros com `gender = "Other"` | **1** |

Estatísticas das numéricas contínuas (dataset completo, antes do split):

| | age | avg_glucose_level | bmi |
|---|---:|---:|---:|
| mínimo | 0,08 | 55,12 | 10,30 |
| mediana | 45,0 | 91,88 | 28,10 |
| máximo | 82,0 | 271,74 | 97,60 |

As escalas são muito diferentes (glicose chega a ~272, idade no máximo a 82) —
esse é o dado concreto que justifica o `MinMaxScaler` na Seção 1.4: sem
normalização, a distância euclidiana/Manhattan do KNN seria dominada pela
coluna de maior magnitude (glicose), e idade/IMC quase não pesariam na decisão.

Um gráfico de barras da distribuição do alvo é salvo em
`resultados/fig_distribuicao_alvo.png`.

## 1.3 Limpeza estrutural (fora do pipeline)

Estas operações **removem linhas/colunas** e não aprendem nenhum parâmetro dos
dados — por isso rodam uma única vez, antes do split, e não entram no
`ColumnTransformer`:

```python
df_limpo = df.drop(columns=["id"])          # id: sem valor preditivo
df_limpo = df_limpo.drop_duplicates()        # 0 duplicatas neste dataset
df_limpo = df_limpo[df_limpo["gender"] != "Other"].reset_index(drop=True)
```

**Decisão sobre `gender = "Other"`:** existe **exatamente 1** registro com esse
valor no dataset inteiro. Uma única instância não permite a nenhum modelo
aprender um padrão para essa categoria, e sua presença quebraria a codificação
binária de duas colunas planejada para `gender` (Seção 1.4). A opção mais
simples e honesta foi remover esse único registro, documentando a perda de
0,02% dos dados — a alternativa (mantê-lo como terceira categoria one-hot)
adicionaria uma coluna quase sempre zero, sem ganho real de informação.

Resultado: `df_limpo` passa de 5.110 para **5.109 linhas** (perde só o registro
`Other`; nenhuma duplicata foi removida neste dataset).

## 1.4 Transformações que aprendem parâmetros (dentro do pipeline)

Diferente da limpeza estrutural, estas transformações **calculam estatísticas
a partir dos dados** (mediana, mínimo/máximo, categorias vistas) — por isso
**não podem** rodar sobre o dataset inteiro antes do split, sob pena de
vazamento de dados (o modelo "veria" estatísticas do conjunto de teste antes
de ser avaliado nele). Elas são encapsuladas em um `ColumnTransformer`,
ajustado (`fit`) **exclusivamente com o conjunto de treino** dentro do
pipeline da Fase 2.

```python
COLUNAS_NUMERICAS      = ["age", "avg_glucose_level", "bmi"]
COLUNAS_BINARIAS        = ["gender", "ever_married", "Residence_type"]
COLUNAS_NOMINAIS        = ["work_type", "smoking_status"]
COLUNAS_JA_BINARIAS     = ["hypertension", "heart_disease"]

def construir_preprocessador() -> ColumnTransformer:
    rota_numericas = PipelineSklearn(steps=[
        ("imputador_mediana", SimpleImputer(strategy="median")),
        ("normalizador", MinMaxScaler()),
    ])
    return ColumnTransformer(transformers=[
        ("num", rota_numericas, COLUNAS_NUMERICAS),
        ("bin", OneHotEncoder(drop="if_binary", sparse_output=False), COLUNAS_BINARIAS),
        ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), COLUNAS_NOMINAIS),
        ("direto", "passthrough", COLUNAS_JA_BINARIAS),
    ], remainder="drop")
```

### a) Numéricas contínuas — mediana + MinMax

- **Imputação pela mediana** (`SimpleImputer(strategy="median")`) preenche os
  201 nulos de `bmi`. A mediana foi escolhida por ser **robusta a outliers**:
  o IMC tem uma cauda longa (máximo 97,6 contra uma mediana de 28,1), então a
  média seria puxada para cima por poucos valores extremos. Remover as linhas
  foi descartado por eliminar ~4% dos dados, incluindo casos raros de AVC.
- **MinMaxScaler** reescala cada coluna para o intervalo **[0, 1]**. É
  crítico para o KNN, que decide por distância entre pacientes.

Parâmetros efetivamente aprendidos pelo pipeline vencedor (ajustados **apenas
com o conjunto de treino**, 4.087 pacientes):

| Coluna | mediana (imputação) | mínimo (treino) | máximo (treino) |
|---|---:|---:|---:|
| `age` | 45,00 | 0,08 | 82,00 |
| `avg_glucose_level` | 91,89 | 55,22 | 271,74 |
| `bmi` | 28,10 | 10,30 | 97,60 |

(O mínimo de `avg_glucose_level` no treino, 55,22, difere ligeiramente do
mínimo do dataset completo, 55,12 — o menor valor absoluto caiu no conjunto de
teste. Isso é esperado e é exatamente por isso que o scaler deve ser ajustado
só no treino: no `app.py`, um paciente com glicose abaixo de 55,22 geraria um
valor normalizado ligeiramente negativo, o que o `MinMaxScaler` permite sem
erro.)

### b) Categóricas binárias — codificação 0/1

`OneHotEncoder(drop="if_binary")` sobre `gender`, `ever_married`,
`Residence_type`: com exatamente duas categorias, o `drop="if_binary"`
mantém **uma única coluna 0/1** (equivalente a Label Encoding, mas com o
mapeamento guardado dentro do pipeline, e não codificado à mão). Categorias
aprendidas e coluna resultante:

| Coluna original | Categorias vistas no treino | Coluna gerada |
|---|---|---|
| `gender` | Female, Male | `gender_Male` (Female = referência/0) |
| `ever_married` | No, Yes | `ever_married_Yes` (No = referência/0) |
| `Residence_type` | Rural, Urban | `Residence_type_Urban` (Rural = referência/0) |

### c) Categóricas nominais — One-Hot Encoding completo

`OneHotEncoder(handle_unknown="ignore")` sobre `work_type` (5 categorias) e
`smoking_status` (4 categorias) → **9 colunas** binárias. Aqui o Label Encoding
foi descartado deliberadamente: numerar as categorias (ex.: Private=3,
Govt_job=1) criaria uma ordem artificial que tanto o KNN (distância) quanto a
Árvore (limiares de divisão) interpretariam como magnitude — uma diferença que
não existe na realidade entre tipos de trabalho ou status de fumante.

`handle_unknown="ignore"` faz a interface (`app.py`) não quebrar se um dado
novo trouxer uma categoria fora das vistas no treino: a linha vira zeros nas
colunas one-hot daquela variável, em vez de lançar exceção.

Categorias aprendidas no treino:
- `work_type`: Govt_job, Never_worked, Private, Self-employed, children
- `smoking_status`: Unknown, formerly smoked, never smoked, smokes

### d) Já binárias — passagem direta

`hypertension` e `heart_disease` já nascem como 0/1 no CSV original;
`"passthrough"` as inclui sem qualquer transformação.

## 1.5 Resultado do pré-processamento

Ao final, cada paciente é representado por **17 variáveis numéricas**, todas
em escala comparável (a maioria em [0, 1]):

```
num__age, num__avg_glucose_level, num__bmi,
bin__gender_Male, bin__ever_married_Yes, bin__Residence_type_Urban,
nom__work_type_Govt_job, nom__work_type_Never_worked, nom__work_type_Private,
nom__work_type_Self-employed, nom__work_type_children,
nom__smoking_status_Unknown, nom__smoking_status_formerly smoked,
nom__smoking_status_never smoked, nom__smoking_status_smokes,
direto__hypertension, direto__heart_disease
```

(3 numéricas + 3 binárias + 9 nominais + 2 já-binárias = 17.) Este vetor de 17
posições é o que efetivamente entra no SMOTE (Fase 2) e nos classificadores.

### Atenção: encoding e balanceamento são dois problemas diferentes

É comum confundir estas duas etapas, mas elas resolvem problemas
**independentes**, em pontos diferentes do pipeline:

| | Encoding (esta seção, Fase 1) | SMOTE (Fase 2) |
|---|---|---|
| **Problema que resolve** | KNN e Árvore só operam sobre números — texto (`"Private"`, `"Female"`) não pode entrar direto na conta de distância do KNN nem no limiar de corte da Árvore. | Apenas 4,9% dos pacientes têm `stroke = 1` — um modelo treinado direto aprende que "prever sempre sem AVC" já minimiza o erro médio. |
| **O que muda** | O **formato** de cada valor (texto → número), sem alterar quantas linhas existem nem a proporção de classes. | A **quantidade de exemplos** da classe minoritária no treino (por meio de exemplos sintéticos), sem alterar o significado de nenhuma coluna. |
| **Em que dados age** | Todas as colunas de entrada (`X`), em qualquer linha do dataset. | Só o conjunto de treino, e só depois do split — nunca a coluna-alvo isolada de `X`. |

Ou seja: **converter texto em número não resolve o desbalanceamento, e balancear as classes não dispensa a conversão de texto em número** — o dataset passa pelas duas etapas, nesta ordem, porque cada uma ataca uma limitação distinta dos algoritmos. Se só o encoding fosse feito (sem SMOTE), os modelos ainda aprenderiam majoritariamente a classe 0; se o SMOTE fosse aplicado antes do encoding, ele tentaria sintetizar exemplos interpolando texto, o que não é matematicamente definido — por isso o SMOTE só pode entrar **depois** do `ColumnTransformer` no pipeline (Seção 1.6).

## 1.6 Por que isso fica num `ColumnTransformer` e não em código solto

Encapsular tudo em um objeto `ColumnTransformer` (em vez de, por exemplo,
chamar `pd.get_dummies` e `MinMaxScaler` soltos no notebook) tem três efeitos
práticos que sustentam as fases seguintes:

1. **Sem vazamento de dados** — o `fit` (aprendizado da mediana, do
   min/max, das categorias) só acontece dentro do laço de treino (Fase 2),
   sobre `X_treino`; o `X_teste` só passa por `transform`.
2. **Reprodutibilidade** — qualquer combinação de KNN/Árvore usa exatamente o
   mesmo pré-processamento, construído pela função `construir_preprocessador()`.
3. **Portabilidade para a interface** — como o pré-processador vira o primeiro
   passo do `Pipeline` salvo em `.joblib` (Fase Bônus), o `app.py` nunca
   precisa reimplementar mediana, escala ou encoding: envia os 10 campos
   brutos do formulário e recebe a predição.
