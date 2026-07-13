# Fase 3 — Avaliação de Métricas, Comparação e Seleção do Melhor Modelo

> Código-fonte: `treino_modelo.py`, seções 5, 6 e 7 (linhas 386–607).
> Artefatos gerados: `resultados/resultados_comparativos.csv`,
> `resultados/resultados_efeito_smote.csv`, `resultados/fig_matrizes_knn.png`,
> `resultados/fig_matrizes_arvore.png`, `modelos/modelo_avc.joblib`.

Esta é a fase de maior peso na avaliação do projeto (40%). Ela parte dos 10
modelos treinados na Fase 2 e responde três perguntas: qual métrica decide o
"melhor" modelo, o que cada modelo realmente acertou/errou, e por que a
acurácia sozinha enganaria aqui.

## 3.1 Por que a acurácia é enganosa neste problema (medido, não hipotético)

Antes de comparar os 10 modelos, o script calcula um **baseline trivial**:
um "modelo" que prevê "sem AVC" para todos os 1.022 pacientes de teste —
zero lógica, apenas a classe majoritária:

```python
acuracia_baseline = (y_teste == 0).mean()   # 972 / 1022 = 0,9511
```

| Métrica | Baseline trivial |
|---|---:|
| Acurácia | **95,11%** |
| Recall (AVC) | **0%** |

O baseline supera em acurácia **todos os 10 modelos treinados** — e ainda
assim é clinicamente inútil: não identifica um único paciente de risco. Isso
comprova, com o próprio dataset, por que a acurácia não pode ser o critério
de decisão em um problema com 95% dos exemplos em uma classe: ela mede
sobretudo a capacidade de acertar a classe majoritária, não a capacidade de
detectar a classe rara que importa clinicamente.

## 3.2 Métrica decisória: Recall da classe AVC

Definições aplicadas ao contexto (a partir da matriz de confusão 2×2, com
foco sempre na classe `stroke = 1`):

- **Recall (AVC)** = VP / (VP + FN) — dos pacientes que **de fato** tiveram
  AVC, quantos o modelo detectou. Seu complemento, o falso negativo, é
  **mandar para casa, sem acompanhamento, um paciente que terá AVC**.
- **Precisão (AVC)** = VP / (VP + FP) — dos pacientes **apontados como
  risco**, quantos realmente tiveram AVC. Seu complemento, o falso positivo,
  custa exames adicionais — caro, mas não fatal.
- **F1-Score** = média harmônica de precisão e recall — usada como
  **critério de desempate**.
- **Acurácia** = (VP+VN)/total — reportada por completude, nunca usada para
  decidir (Seção 3.1).

**Decisão de projeto:** o melhor modelo é o de maior Recall (AVC), com
empates resolvidos pelo F1-Score. Justificativa clínica: entre os dois erros
possíveis, o falso negativo é sistematicamente mais grave que o falso
positivo neste domínio.

## 3.3 Tabela comparativa consolidada (números reais do conjunto de teste)

Gerada e ordenada pela métrica decisória:

```python
tabela_resultados = (
    pd.DataFrame(resultados)[COLUNAS_METRICAS]
    .sort_values("Recall (AVC)", ascending=False)
    .reset_index(drop=True)
)
tabela_resultados.to_csv(PASTA_RESULTADOS / "resultados_comparativos.csv", index=False)
```

| Modelo | Acurácia | Precisão (AVC) | Recall (AVC) | F1-Score (AVC) |
|---|---:|---:|---:|---:|
| **Árvore (max_depth=5)** ← melhor | 0,6840 | 0,1155 | **0,8200** | **0,2025** |
| Árvore (max_depth=3) | 0,6605 | 0,1082 | 0,8200 | 0,1911 |
| Árvore (max_depth=10) | 0,7407 | 0,1147 | 0,6400 | 0,1945 |
| KNN (K=7, Manhattan) | 0,8366 | 0,1274 | 0,4000 | 0,1932 |
| KNN (K=7, Euclidiana) | 0,8131 | 0,1061 | 0,3800 | 0,1659 |
| KNN (K=5, Euclidiana) | 0,8278 | 0,1063 | 0,3400 | 0,1619 |
| KNN (K=5, Manhattan) | 0,8503 | 0,1241 | 0,3400 | 0,1818 |
| KNN (K=3, Euclidiana) | 0,8552 | 0,1048 | 0,2600 | 0,1494 |
| Árvore (max_depth=None) | 0,8718 | 0,1143 | 0,2400 | 0,1548 |
| KNN (K=3, Manhattan) | 0,8630 | 0,0982 | 0,2200 | 0,1358 |
| Baseline trivial | 0,9511 | 0,0000 | 0,0000 | 0,0000 |

## 3.4 Matrizes de confusão como heatmaps

### Convenção adotada (código, linhas 424–431)

```
                Previsto: Sem AVC   Previsto: AVC
Real: Sem AVC        VN                  FP
Real: AVC            FN                  VP
```

Linhas = classe real, colunas = classe prevista (convenção padrão do
scikit-learn, `confusion_matrix(y_teste, y_previsto, labels=[0, 1])`). A
célula inferior-esquerda (FN) é sempre o número clinicamente mais sensível
do gráfico.

### Escolha de cor: por que normalizar por linha

```python
proporcao_linha = matriz / matriz.sum(axis=1, keepdims=True)
sns.heatmap(proporcao_linha, cmap=CMAP_AZUL, vmin=0, vmax=1, ...)
```

A cor de cada célula usa a **proporção dentro da classe real** (linha), não a
contagem absoluta. Com 972 negativos contra apenas 50 positivos no teste, uma
escala de cor por contagem absoluta faria a linha "AVC" inteira parecer quase
branca — os 41 verdadeiros positivos ficariam visualmente invisíveis ao lado
das centenas de verdadeiros negativos. Normalizando por linha, a diagonal
mostra o Recall de cada classe diretamente na intensidade da cor. A anotação
de texto em cada célula mostra as duas informações (contagem absoluta e
percentual da linha), preservando o dado bruto por trás da normalização.

A paleta é uma única rampa sequencial em azul (claro→escuro) — cor sequencial
para codificar magnitude; deliberadamente **não** se usa um mapa de cores
multicolorido (arco-íris), que sugeriria categorias distintas em vez de uma
grandeza contínua.

### Matriz de confusão do modelo vencedor

Árvore de Decisão, `max_depth=5`, 1.022 pacientes de teste:

| | Previsto: Sem AVC | Previsto: AVC |
|---|---:|---:|
| **Real: Sem AVC** (972) | 658 (VN, 67,7%) | 314 (FP, 32,3%) |
| **Real: AVC** (50) | **9 (FN, 18,0%)** | **41 (VP, 82,0%)** |

Leitura: o modelo captura 41 dos 50 casos reais de AVC — os 9 que escapam são
o custo residual aceito. O preço é o excesso de cautela: 314 pacientes
saudáveis sinalizados como risco (Precisão = 11,55%, ou seja, de cada ~9
alertas, 1 se confirma). Em um fluxo de **triagem** — o alerta leva a exame
adicional, não a um diagnóstico definitivo — esse é o trade-off desejado: o
custo de 314 falsos positivos é operacional; o custo de deixar 9 (ou mais)
falsos negativos passarem sem acompanhamento é, potencialmente, uma vida.

Os heatmaps completos das 10 combinações estão em
`resultados/fig_matrizes_knn.png` (2×3: KNN) e
`resultados/fig_matrizes_arvore.png` (2×2: Árvore).

## 3.5 Ablação: o que acontece sem o SMOTE

Para comprovar — não apenas argumentar — que o SMOTE é indispensável, o
script retreina a melhor configuração de cada algoritmo **sem** o passo SMOTE
(mesmo `ColumnTransformer`, mesmo split), usando o `Pipeline` padrão do
scikit-learn:

```python
pipeline_sem_smote = PipelineSklearn(steps=[
    ("preprocessamento", construir_preprocessador()),
    ("modelo", modelo_base.__class__(**modelo_base.get_params())),
])
```

| Modelo | Cenário | Recall (AVC) | Precisão (AVC) | Acurácia |
|---|---|---:|---:|---:|
| KNN (K=7, Manhattan) | com SMOTE | 0,40 | 0,1274 | 0,8366 |
| KNN (K=7, Manhattan) | **sem SMOTE** | **0,02** | 0,5000 | 0,9511 |
| Árvore (max_depth=5) | com SMOTE | 0,82 | 0,1155 | 0,6840 |
| Árvore (max_depth=5) | **sem SMOTE** | **0,04** | 0,4000 | 0,9501 |

Sem balanceamento, os dois algoritmos degeneram para perto do baseline
trivial: acurácia ~95%, mas Recall de apenas 2–4%. A precisão sem SMOTE
parece "melhor" (0,40–0,50), mas é enganosa pela mesma razão da Seção 3.1: com
tão poucas previsões positivas, as raras que acertam elevam artificialmente a
precisão, enquanto a imensa maioria dos casos reais de AVC passa despercebida.
**Conclusão do projeto: o SMOTE não é um refinamento opcional — é a técnica
que torna o problema tratável** dado o desbalanceamento de 95%/5%.

## 3.6 Interpretabilidade — importância de variáveis na Árvore vencedora

Vantagem exclusiva da Árvore (o KNN não expõe por que decide): o atributo
`feature_importances_` mede quanto cada variável contribuiu para reduzir a
impureza Gini nas divisões:

| Variável | Importância |
|---|---:|
| `age` | 0,8853 |
| `gender_Male` | 0,0396 |
| `avg_glucose_level` | 0,0326 |
| `bmi` | 0,0287 |
| `hypertension` | 0,0126 |
| `work_type_Govt_job` | 0,0010 |
| `work_type_Self-employed` | 0,0002 |
| (demais 9 variáveis) | 0,0000 |

A idade domina isoladamente (88,5% da decisão) — coerente com o fator de
risco clínico mais conhecido de AVC. As 9 colunas de `work_type`/`smoking_status`
com importância zero mostram que, nesta árvore rasa (profundidade 5, 27
folhas), poucas variáveis chegam a ser usadas — efeito esperado quando uma
única variável (idade) já separa bem a maior parte dos casos.

## 3.7 Seleção do melhor modelo e desempate

```python
melhor = max(candidatos, key=lambda r: (r["Recall (AVC)"], r["F1-Score (AVC)"]))
```

A chave de ordenação usa uma tupla: primeiro o Recall (critério oficial),
depois o F1 (desempate). `max_depth=3` e `max_depth=5` empatam em Recall
(0,8200, mesmos 9 falsos negativos) — o desempate por F1-Score
(0,2025 vs 0,1911) favorece **`max_depth=5`**, que comete 24 falsos positivos
a menos (314 vs 338) para o mesmo poder de detecção: estritamente melhor no
trade-off Precisão/Recall.

**Modelo eleito: Árvore de Decisão, `max_depth=5`.**

## 3.8 Salvamento do pipeline vencedor

```python
joblib.dump(melhor["pipeline"], CAMINHO_MODELO)   # modelos/modelo_avc.joblib
```

O objeto salvo é o **pipeline completo já treinado** (pré-processamento +
SMOTE + Árvore) extraído diretamente do dicionário de resultados da Fase 2 —
não há retreino nem reconstrução: é literalmente o mesmo objeto que gerou as
métricas desta seção, garantindo rastreabilidade total entre o relatório e o
artefato consumido pela interface (Fase Bônus).

O script encerra com uma simulação do fluxo real: recarrega o `.joblib` do
disco e envia um paciente fictício com dados **brutos** (mesmo formato que o
formulário do `app.py` produzirá), confirmando que o pipeline recarregado
reaplica sozinho toda a Fase 1 sem qualquer código adicional.
