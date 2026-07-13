# Fase 3 — Avaliação de Métricas, Comparação e Seleção dos Modelos Finais

> Código-fonte: `treino_modelo.py`, seções 5, 6 e 7.
> Artefatos gerados: `resultados/resultados_comparativos.csv`,
> `resultados/resultados_efeito_smote.csv`, `resultados/fig_matrizes_knn_uniforme.png`,
> `resultados/fig_matrizes_knn_distancia.png`, `resultados/fig_matrizes_arvore.png`,
> `modelos/modelo_knn.joblib`, `modelos/modelo_arvore.joblib`.

Esta é a fase de maior peso na avaliação do projeto (40%). Ela parte dos 40
modelos treinados na Fase 2 (24 KNN + 16 Árvores) e responde três perguntas:
qual métrica decide o "melhor" modelo, o que cada modelo realmente
acertou/errou, e por que a acurácia sozinha enganaria aqui.

## 3.1 Por que a acurácia é enganosa neste problema (medido, não hipotético)

Antes de comparar os 40 modelos, o script calcula um **baseline trivial**:
um "modelo" que prevê "sem AVC" para todos os 1.022 pacientes de teste —
zero lógica, apenas a classe majoritária:

```python
acuracia_baseline = (y_teste == 0).mean()   # 972 / 1022 = 0,9511
```

| Métrica | Baseline trivial |
|---|---:|
| Acurácia | **95,11%** |
| Recall (AVC) | **0%** |

O baseline supera em acurácia **todos os 40 modelos treinados** — e ainda
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

Com 40 modelos, a tabela abaixo mostra a Árvore recomendada, a configuração
de maior Recall do grid completo (que **não** foi adotada — ver observação
abaixo), o melhor KNN (**K=15, Euclidiana, peso uniforme**, resultado da
ampliação de range discutida na Fase 2) e os demais destaques; a lista
completa das 40 combinações está em `resultados/resultados_comparativos.csv`.

| Modelo | Acurácia | Precisão (AVC) | Recall (AVC) | F1-Score (AVC) |
|---|---:|---:|---:|---:|
| Árvore (max_depth=3, max_features=sqrt) — maior Recall do grid, **não adotada** (Seção 3.6) | 0,4384 | 0,0733 | 0,9000 | 0,1355 |
| **Árvore (max_depth=5, max_features=todas)** ← recomendada | 0,6840 | 0,1155 | **0,8200** | **0,2025** |
| Árvore (max_depth=3, max_features=todas) | 0,6605 | 0,1082 | 0,8200 | 0,1911 |
| Árvore (max_depth=5, max_features=50%) | 0,7074 | 0,1073 | 0,6800 | 0,1853 |
| Árvore (max_depth=10, max_features=todas) | 0,7407 | 0,1147 | 0,6400 | 0,1945 |
| **KNN (K=15, Euclidiana, peso uniforme)** ← melhor KNN | 0,7583 | 0,1044 | **0,5200** | 0,1739 |
| KNN (K=15, Manhattan, peso uniforme) | 0,7926 | 0,1143 | 0,4800 | 0,1846 |
| KNN (K=15, Euclidiana, peso por distância) | 0,7965 | 0,1089 | 0,4400 | 0,1746 |
| KNN (K=11, Euclidiana, peso uniforme) | 0,7818 | 0,1014 | 0,4400 | 0,1648 |
| KNN (K=9, Euclidiana, peso uniforme) | 0,7916 | 0,1024 | 0,4200 | 0,1647 |
| KNN (K=7, Manhattan, peso uniforme) | 0,8366 | 0,1274 | 0,4000 | 0,1932 |
| ⋮ (demais combinações de KNN e Árvore, Recall entre 0,20 e 0,40) | | | | |
| Árvore (max_depth=None, max_features=todas) | 0,8718 | 0,1143 | 0,2400 | 0,1548 |
| KNN (K=3, Manhattan, peso por distância) | 0,8699 | 0,0971 | 0,2000 | 0,1307 |
| Baseline trivial | 0,9511 | 0,0000 | 0,0000 | 0,0000 |

> A configuração de maior Recall nominal (`max_depth=3, max_features=sqrt`,
> 90%) sinaliza 569 dos 972 pacientes saudáveis do teste como risco (58,5% de
> falso-alarme) — inviável como recomendação de triagem. A Seção 3.6 explica
> por que a eleição do melhor modelo é restrita à família
> `max_features="todas"`; a investigação completa está em
> `docs/nota-max-features-idade.md`.

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

Árvore de Decisão, `max_depth=5, max_features=todas` (modelo recomendado),
1.022 pacientes de teste:

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

Os heatmaps completos das 40 combinações estão em
`resultados/fig_matrizes_knn_uniforme.png` (2×6: KNN, peso uniforme),
`resultados/fig_matrizes_knn_distancia.png` (2×6: KNN, peso por distância) e
`resultados/fig_matrizes_arvore.png` (4×4: Árvore — linhas = `max_depth`,
colunas = `max_features`). Os dois heatmaps de KNN são separados por peso
porque uma única grade com as 24 combinações ficaria ilegível; dentro de cada
figura, as linhas são as 2 métricas de distância e as colunas os 6 valores de
K, na mesma ordem do laço de treino.

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
| KNN (K=15, Euclidiana, peso uniforme) | com SMOTE | 0,52 | 0,1044 | 0,7583 |
| KNN (K=15, Euclidiana, peso uniforme) | **sem SMOTE** | **0,04** | 1,0000 | 0,9530 |
| Árvore (max_depth=5) | com SMOTE | 0,82 | 0,1155 | 0,6840 |
| Árvore (max_depth=5) | **sem SMOTE** | **0,04** | 0,4000 | 0,9501 |

Sem balanceamento, os dois algoritmos degeneram para perto do baseline
trivial: acurácia ~95%, mas Recall de apenas 4%. A precisão sem SMOTE parece
"melhor" (0,40–1,00), mas é enganosa pela mesma razão da Seção 3.1: com tão
poucas previsões positivas, as raras que acertam elevam artificialmente a
precisão — o KNN sem SMOTE chega a 100% de precisão prevendo AVC para
praticamente ninguém —, enquanto a imensa maioria dos casos reais de AVC
passa despercebida. **Conclusão do projeto: o SMOTE não é um refinamento
opcional — é a técnica que torna o problema tratável** dado o
desbalanceamento de 95%/5%.

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

Essa concentração motivou uma investigação à parte: usar `max_features` para
forçar a árvore a considerar outras variáveis em cada divisão (Fase 2, Seção
2.3). A técnica funcionou — reduziu a importância de `age` para 20%–66% no
restante do grid —, mas a configuração de maior Recall resultante
(`max_depth=3, max_features=sqrt`) sinaliza 58,5% dos pacientes saudáveis
como risco, contra 32,3% da árvore original. Por isso essa família de
configurações **não entra na eleição do melhor modelo** (Seção 3.7) — fica
registrada como experimento em `docs/nota-max-features-idade.md`.

## 3.7 Seleção do melhor KNN, da melhor Árvore, e desempate

O sistema final disponibiliza **dois modelos** ao usuário (Seção 4 — Fase
Bônus), não apenas um. Por isso a seleção ocorre **dentro de cada família de
hiperparâmetros**, com o mesmo critério (Recall, desempate por F1):

```python
melhor_knn = max(resultados_knn, key=lambda r: (r["Recall (AVC)"], r["F1-Score (AVC)"]))

# Restrita a max_features="todas": ver Seção 3.6 e docs/nota-max-features-idade.md
# sobre por que a família max_features não entra na eleição oficial.
resultados_arvore_familia_original = [
    r for r in resultados_arvore if "max_features=todas" in r["Modelo"]
]
melhor_arvore = max(
    resultados_arvore_familia_original, key=lambda r: (r["Recall (AVC)"], r["F1-Score (AVC)"])
)
```

Dentro da família original da Árvore (`max_features="todas"`), `max_depth=3`
e `max_depth=5` empatam em Recall (0,8200, mesmos 9 falsos negativos) — o
desempate por F1-Score (0,2025 vs 0,1911) favorece **`max_depth=5`**, que
comete 24 falsos positivos a menos (314 vs 338) para o mesmo poder de
detecção: estritamente melhor no trade-off Precisão/Recall. Dentro da família
KNN, **`K=15`, distância Euclidiana, peso uniforme** venceu as demais 23
combinações (Recall 0,52, Seção 3.3) — o resultado da segunda rodada de
exploração descrita na Fase 2, que ampliou o range de K e testou o parâmetro
`weights`.

Os dois modelos ficam então disponíveis na interface. Para a **recomendação
clínica** do relatório, o script ainda compara os dois vencedores entre si
pelo mesmo critério:

```python
recomendado = max([melhor_knn, melhor_arvore],
                  key=lambda r: (r["Recall (AVC)"], r["F1-Score (AVC)"]))
```

**Recomendada: Árvore de Decisão, `max_depth=5, max_features=todas`** (Recall
0,82 contra 0,52 do melhor KNN) — mas o KNN permanece acessível na interface
como alternativa.

## 3.8 Salvamento dos dois pipelines

```python
joblib.dump(melhor_knn["pipeline"], CAMINHO_MODELO_KNN)       # modelos/modelo_knn.joblib
joblib.dump(melhor_arvore["pipeline"], CAMINHO_MODELO_ARVORE) # modelos/modelo_arvore.joblib
```

Cada objeto salvo é o **pipeline completo já treinado** (pré-processamento +
SMOTE + classificador) extraído diretamente do dicionário de resultados da
Fase 2 — não há retreino nem reconstrução: são literalmente os mesmos objetos
que geraram as métricas desta seção, garantindo rastreabilidade total entre o
relatório e os artefatos consumidos pela interface (Fase Bônus).

O script encerra com uma simulação do fluxo real para **os dois modelos**:
recarrega cada `.joblib` do disco e envia um paciente fictício com dados
**brutos** (mesmo formato que o formulário do `app.py` produzirá),
confirmando que cada pipeline recarregado reaplica sozinho toda a Fase 1 sem
qualquer código adicional.
