# Nota técnica: por que o SMOTENC reduziu o Recall (e por que o projeto ficou com o SMOTE)

> Este documento registra uma investigação feita durante o desenvolvimento do
> projeto: uma tentativa de trocar `SMOTE` por `SMOTENC` para corrigir um bug
> real, e o motivo pelo qual essa troca foi **revertida**. Serve como registro
> da decisão, não como descrição do pipeline final — para o pipeline
> efetivamente em uso, ver `treino_modelo.py` e `docs/relatorio.md`.

## 1. O bug que motivou a tentativa

Durante testes manuais do app, foi observado que **pacientes com hipertensão
ou doença cardíaca recebiam probabilidade de AVC *menor*** que pacientes sem
essas condições, em alguns perfis — uma inversão sem sentido clínico.

Investigando a Árvore vencedora da época (`max_depth=5`, treinada com o
`SMOTE` comum), a causa raiz identificada foi: `hypertension` e
`heart_disease` são variáveis binárias (0/1), mas o `SMOTE` interpola
**linearmente todas as colunas** entre dois vizinhos da classe minoritária —
inclusive essas. Isso gera pacientes sintéticos com valores como
`hypertension = 0,37`, que não existem na realidade. Medimos diretamente:
**~7% dos valores sintéticos de `hypertension`** e **~5% de `heart_disease`**
ficavam fora de `{0, 1}` no treino balanceado.

A árvore, por sua vez, aprendeu a explorar esse ruído: sua estrutura usava
`hypertension` em **três limiares diferentes**:

```
hypertension <= 0.01
hypertension >  0.01
    hypertension <= 1.00
    hypertension >  1.00      <- limiar impossível para um paciente real
        hypertension <= 0.50
        hypertension >  0.50
```

O nó `hypertension > 1.00` só existe porque valores sintéticos podiam
extrapolar ligeiramente acima de 1 (artefato numérico da interpolação), e o
conjunto desses três cortes acabou empurrando alguns perfis de paciente para
o lado errado da árvore — a inversão observada no app.

## 2. A tentativa de correção: SMOTENC

O `SMOTENC` (SMOTE for Nominal and Continuous) resolve exatamente esse tipo de
problema: para as colunas marcadas como categóricas, ele substitui a
interpolação linear por **voto de maioria entre os vizinhos**, preservando os
valores discretos originais em todo exemplo sintético. Aplicado ao pipeline
(14 das 17 colunas do espaço transformado marcadas como categóricas —
`hypertension`, `heart_disease` e todas as dummies one-hot), o bug foi
**confirmado corrigido**: `np.unique` sobre `hypertension` pós-balanceamento
passou a retornar exatamente `[0. 1.]`, sem frações, e o split
`hypertension > 1.00` desapareceu.

## 3. O efeito colateral: o Recall caiu

| Modelo | Recall com SMOTE | Recall com SMOTENC |
|---|---:|---:|
| Árvore (max_depth=3) | 0,82 (empate) | **0,66** |
| Árvore (max_depth=5) | 0,82 (empate, vencedora) | 0,58 |
| Árvore (max_depth=10) | 0,64 | 0,50 |
| Árvore (max_depth=None) | 0,24 | 0,28 |
| Melhor KNN | 0,52 (K=15, Euclidiana, uniforme) | **0,56** (K=15, Manhattan, uniforme) |

A Árvore (o modelo recomendado do projeto) perdeu 16 a 24 pontos percentuais
de Recall dependendo da profundidade; o KNN, na direção oposta, melhorou
ligeiramente.

## 4. Por que o Recall da Árvore caiu — mecanismo identificado

Comparando a importância de variáveis antes e depois:

| Variável | Importância (SMOTE, `max_depth=5`) | Importância (SMOTENC, `max_depth=3`) |
|---|---:|---:|
| `age` | 0,8853 | 0,9315 |
| `gender_Male` | 0,0396 | 0,0144 |
| `avg_glucose_level` | 0,0326 | 0,0000 |
| `bmi` | 0,0287 | 0,0233 |
| `hypertension` | 0,0126 | **0,0000** |
| `heart_disease` | 0,0000 | 0,0302 |

**`hypertension` caiu de 3 nós de decisão distintos para zero uso na árvore
inteira.** Isto é a explicação central: com o `SMOTE` comum, a corrupção
fracionária transformava uma variável binária em algo que se comportava como
**contínua**, dando à árvore rasa (limitada a poucos níveis) **múltiplos
pontos de corte utilizáveis** na mesma variável (0,01 / 0,50 / 1,00) para
reduzir a impureza Gini em vários níveis da árvore ao mesmo tempo — uma
flexibilidade extra que uma variável genuinamente binária nunca ofereceria
(só existe um corte possível: 0,5).

Essa flexibilidade extra aparentemente ajudava — por acaso, não por sinal
clínico real — a capturar mais dos 50 casos positivos do conjunto de teste.
Um indício adicional: **sob SMOTE, aumentar a profundidade de 3 para 5 não
piorava o Recall** (ambas empatavam em 0,82) — sinal de que a árvore não
precisava "gastar" profundidade extra de forma prejudicial, porque já tinha
graus de liberdade suficientes vindos do ruído em `hypertension`. **Sob
SMOTENC, o Recall já cai da profundidade 3 para a 5** (0,66 → 0,58) — sem essa
muleta, cada nível adicional de profundidade tende a ajustar ruído genuíno
(a própria inversão de `heart_disease` documentada em `docs/relatorio.md`,
Seção 5.5, é um exemplo desse ajuste a ruído, não a sinal).

Em resumo: **uma fração do Recall de 82% do modelo original não vinha de
sinal clínico, e sim de a árvore explorar uma inconsistência do balanceamento
sintético.** O SMOTENC removeu essa inconsistência — e removeu, junto, o
Recall que ela emprestava artificialmente ao modelo.

### Por que o KNN se moveu na direção oposta

O KNN também é afetado pela mesma corrupção, mas de forma diferente: as
colunas one-hot e binárias, quando fracionadas pelo SMOTE, funcionam como
dimensões contínuas extras e artificiais no cálculo de distância — o que
pode distorcer levemente a noção de "vizinho mais próximo" em relação ao que
seria calculado com vetores binários limpos. Com o SMOTENC preservando os
valores discretos reais, as distâncias passam a refletir melhor a
similaridade genuína entre pacientes, o que aparentemente ajudou (ainda que
marginalmente: +4 pontos percentuais) o melhor KNN.

## 5. Decisão

Optou-se por **manter o `SMOTE` comum** no pipeline de treino, priorizando o
Recall da classe AVC — a métrica decisória do projeto (`docs/relatorio.md`,
Seção 3.4) — sobre a correção do bug de inversão em `hypertension`/
`heart_disease`. Essa decisão foi tomada cientes do trade-off:

- **Ganho ao manter o SMOTE:** Recall de 82% em vez de 66% na Árvore
  recomendada — uma diferença clinicamente relevante (33 vs 41 casos de AVC
  detectados em 50).
- **Custo aceito:** o bug de inversão em variáveis binárias permanece latente
  no modelo — pacientes com hipertensão ou doença cardíaca podem, em alguns
  perfis específicos, receber probabilidade de AVC menor do que deveriam.
  Isso deveria continuar documentado como limitação conhecida em qualquer
  relatório ou apresentação que use o modelo treinado com SMOTE comum.

### Alternativa não testada, para referência futura

Uma forma de obter os dois benefícios ao mesmo tempo — Recall alto **e**
sem a inversão — seria manter o `SMOTENC` (que já resolve a causa raiz do
bug) e usar o parâmetro `monotonic_cst` do `DecisionTreeClassifier`
(disponível desde o scikit-learn 1.4) para **forçar explicitamente** que
`hypertension` e `heart_disease` só possam aumentar a probabilidade prevista
de AVC. Isso não devolveria ao modelo a flexibilidade artificial que o SMOTE
comum dava, mas poderia compensar parte da perda de Recall observada nesta
análise, por uma via diferente (regularização direcionada, não ruído
sintético). Não foi implementada nem testada nesta investigação.
