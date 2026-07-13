# Nota técnica: tentativa de reduzir a dependência da idade na Árvore (e por que foi revertida)

> Este documento registra uma investigação feita durante o desenvolvimento do
> projeto: uma tentativa de usar `max_features` para tornar a Árvore vencedora
> menos dependente de uma única variável (`age`), e o motivo pelo qual essa
> mudança **não foi adotada** como modelo oficial. Serve como registro da
> decisão, não como descrição do pipeline final — para o pipeline efetivamente
> em uso, ver `treino_modelo.py` e `docs/relatorio.md`.

## 1. O problema que motivou a investigação

Na Árvore vencedora original (`max_depth=5`, sem restrição de `max_features`),
a importância de variáveis é extremamente concentrada:

```
age                       0.8853
gender_Male               0.0396
avg_glucose_level         0.0326
bmi                       0.0287
hypertension               0.0126
```

**88,5% de toda a decisão da árvore vem de uma única coluna.** Isso é uma
preocupação legítima de robustez: um modelo que decide quase exclusivamente
por uma variável é frágil a qualquer ruído/erro nessa coluna específica e
ignora, na prática, quase todo o resto do quadro clínico do paciente
(glicose, hipertensão, tabagismo, doença cardíaca) — o oposto do que se
espera de um sistema de apoio à decisão clínica.

## 2. A tentativa: `max_features`

`max_features` é o parâmetro padrão do `DecisionTreeClassifier` para limitar
quantas variáveis a árvore pode **considerar** em cada divisão individual
(não remove nenhuma coluna do dataset — só restringe o leque de candidatas a
cada nó). Quando a variável dominante não está no subconjunto sorteado
daquele nó, a árvore é forçada a dividir por outra. Adicionamos esse eixo ao
laço manual já existente: `max_depth ∈ {3, 5, 10, None}` × `max_features ∈
{None, sqrt, log2, 0,5}` → 16 configurações (contra as 4 originais).

## 3. Resultado: funcionou, mas com um custo maior do que o esperado

| Configuração | Recall | Precisão | Acurácia | Importância de `age` | Falso-alarme* |
|---|---:|---:|---:|---:|---:|
| `max_depth=5, max_features=todas` (original) | 0,82 | 0,1155 | 0,6840 | 88,5% | 32,3% |
| `max_depth=5, max_features=50%` | 0,68 | 0,1073 | 0,7074 | 65,5% | 29,3% |
| `max_depth=5, max_features=sqrt` | 0,72 | 0,0950 | 0,6507 | 48,3% | — |
| `max_depth=3, max_features=sqrt` | **0,90** | 0,0733 | 0,4384 | **23,6%** | **58,5%** |

*Falso-alarme = falsos positivos ÷ pacientes saudáveis no teste (972).

A técnica funcionou exatamente como projetada: `max_features` reduz a
concentração em `age` de forma consistente e mensurável em todo o grid. A
combinação de maior Recall no grid completo, `max_depth=3, max_features=sqrt`,
chega a **superar** a árvore original em Recall (90% vs. 82%, ou seja, 45 vs.
41 dos 50 casos de AVC detectados) e reduz a importância de `age` para 23,6%.

**O problema está no efeito colateral**, revelado só ao calcular a matriz de
confusão completa (a tabela de métricas por si só não deixa isso óbvio): com
precisão de apenas 7,33%, esse modelo classifica **569 dos 972 pacientes
saudáveis do teste como risco de AVC — 58,5% de falso-alarme**. Um sistema de
triagem que sinaliza mais da metade das pessoas saudáveis como "risco" perde
praticamente todo o valor prático de triagem: na prática, equivale a pouco
mais que marcar quase todo mundo como suspeito.

## 4. Por que o Recall subiu tanto ao restringir `max_features`

Com apenas 3 níveis de profundidade e um subconjunto pequeno de variáveis
candidatas por nó (`sqrt(17) ≈ 4`), a árvore é forçada a usar variáveis
secundárias — `ever_married_Yes` (35,5% de importância nessa configuração) e
`avg_glucose_level` (28,0%) passam a liderar a decisão, com `age` caindo para
terceiro lugar (23,6%). Essas variáveis têm um padrão de corte menos
específico que a idade (que sozinha já separa bem grande parte dos casos),
então a árvore acaba criando regras mais abrangentes — capturando mais
positivos reais, mas também muito mais negativos incorretamente. É o mesmo
mecanismo de qualquer troca Recall↔Precisão, só que aqui empurrado a um
extremo pela combinação de pouca profundidade com poucas variáveis
candidatas por nó.

## 5. Decisão

**A eleição oficial do melhor modelo continua restrita à família original**
(`max_features="todas"`, ou seja, sem essa restrição) — o código em
`treino_modelo.py` filtra explicitamente por essa condição antes de aplicar
o critério decisório (maior Recall, desempate por F1), então o modelo salvo
em `modelos/modelo_arvore.joblib` permanece `max_depth=5, max_features=todas`
(Recall 82%, falso-alarme 32,3%), independentemente de haver, no grid
completo, uma configuração de Recall nominalmente maior.

- **Ganho que se perderia ao trocar:** taxa de falso-alarme quase dobraria
  (32,3% → 58,5%) para um ganho de Recall de apenas 8 pontos percentuais
  (82%→90%, ou seja, 4 pacientes a mais detectados em 50) — um trade-off
  desproporcional para um sistema de triagem.
- **Ganho que se perde ao NÃO trocar:** o modelo entregue continua altamente
  dependente de uma única variável (88,5% de importância em `age`) — essa
  fragilidade de interpretabilidade permanece uma limitação documentada,
  não resolvida.

### Alternativa intermediária, para referência futura

`max_depth=5, max_features=50%` é o ponto do grid mais equilibrado: reduz a
importância de `age` de 88,5% para 65,5% e mantém o falso-alarme próximo do
original (29,3% vs. 32,3%), ao custo de 14 pontos percentuais de Recall
(82%→68%). Não foi adotada por ainda representar uma queda de Recall maior
que o ganho de robustez pareceu justificar nesta rodada, mas é a opção mais
razoável caso o critério de decisão do projeto venha a incorporar,
explicitamente, uma penalidade por concentração de importância (o que hoje
não faz parte do critério oficial — Seção 3.2 do `docs/fase3-avaliacao.md`).
