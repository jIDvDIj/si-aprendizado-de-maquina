# Fase Bônus — Deployment (Interface Streamlit)

> Código-fonte: `app.py` (arquivo inteiro, 160 linhas).
> Depende de: `modelos/modelo_avc.joblib`, gerado pela Fase 3.

## 4.1 Princípio central: separação total entre treino e interface

`app.py` **não contém nenhuma lógica de treino, pré-processamento manual ou
cálculo de métrica** — essa é a exigência de maior peso no critério de
"Qualidade do Código" do projeto. Tudo o que o arquivo faz:

1. carrega o `.joblib` (pipeline completo: pré-processamento + SMOTE + Árvore);
2. coleta os dados **brutos** do paciente por um formulário;
3. entrega esses dados ao pipeline e mostra a predição com sua probabilidade.

A garantia de que o app nunca transforma os dados "do seu próprio jeito" —
uma fonte clássica de bugs quando treino e aplicação são mantidos
separadamente — vem de o próprio pipeline salvo reaplicar, sozinho, a mediana
de imputação, a escala MinMax e o One-Hot Encoding aprendidos no treino
(Fase 1). O `app.py` nunca vê o vetor de 17 variáveis numéricas internamente
usado pelo modelo: ele só manipula os 10 campos brutos do CSV original.

## 4.2 Carregamento do pipeline

```python
CAMINHO_MODELO = Path(__file__).parent / "modelos" / "modelo_avc.joblib"

@st.cache_resource
def carregar_pipeline(caminho: Path):
    return joblib.load(caminho)
```

- O caminho é resolvido **a partir do próprio arquivo** (`__file__`), não do
  diretório de trabalho corrente — o app funciona independentemente de onde o
  comando `streamlit run` é executado.
- `@st.cache_resource` faz o `joblib.load` rodar **uma única vez** por
  processo do servidor: sem cache, o Streamlit reexecutaria o script inteiro
  (recarregando o modelo do disco) a cada interação do formulário.
- Se o arquivo não existir, o app mostra uma mensagem de erro amigável
  apontando o comando de treino (`python treino_modelo.py`) e chama
  `st.stop()` — falha explícita em vez de uma exceção crua.

Uma vez carregado, o app expõe qual algoritmo está de fato em uso, lido
diretamente do objeto (nunca hard-coded no texto da interface):

```python
nome_classificador = type(pipeline.named_steps["modelo"]).__name__
```

## 4.3 Tradução de rótulos na borda da interface

O pipeline foi treinado com as categorias **originais em inglês** do CSV
(`Male`/`Female`, `Private`/`Self-employed`/…, `never smoked`/…). Como a
interface é em português, a tradução acontece através de dicionários
bidirecionais, todos únicos e centralizados no topo do arquivo:

```python
OPCOES_GENERO = {"Feminino": "Female", "Masculino": "Male"}
OPCOES_TRABALHO = {
    "Setor privado": "Private", "Autônomo(a)": "Self-employed",
    "Funcionário(a) público(a)": "Govt_job", "Criança": "children",
    "Nunca trabalhou": "Never_worked",
}
OPCOES_FUMANTE = {
    "Nunca fumou": "never smoked", "Fumou no passado": "formerly smoked",
    "Fuma atualmente": "smokes", "Não informado": "Unknown",
}
# + OPCOES_SIM_NAO, OPCOES_CASADO, OPCOES_RESIDENCIA
```

O usuário só vê as chaves em português (`st.selectbox("Gênero", list(OPCOES_GENERO))`);
o valor em inglês é resolvido apenas no momento de montar os dados enviados
ao pipeline. Isso mantém o modelo isolado de qualquer decisão de UI — trocar
o idioma da interface no futuro não exigiria retreinar nada.

## 4.4 Formulário

```python
with st.form("formulario_paciente"):
    ...
    enviado = st.form_submit_button("🩺 Calcular risco", use_container_width=True)
```

Os 10 campos do formulário espelham exatamente as 10 colunas brutas de
entrada do pipeline (todas as colunas do CSV original, exceto `id` e
`stroke`). Usar `st.form` (em vez de widgets soltos) evita que o Streamlit
reprocesse a página inteira a cada campo alterado — a predição só roda
quando o usuário clica em "Calcular risco", reduzindo chamadas desnecessárias
ao pipeline.

## 4.5 Predição e apresentação do resultado

```python
dados_paciente = {
    "gender": OPCOES_GENERO[genero], "age": float(idade), ...
}
paciente = pd.DataFrame([dados_paciente])   # DataFrame de 1 linha, dados BRUTOS

predicao = int(pipeline.predict(paciente)[0])
probabilidade_avc = float(pipeline.predict_proba(paciente)[0][1])
```

- `pd.DataFrame([dados_paciente])` cria uma linha com os mesmos nomes de
  coluna que o `ColumnTransformer` da Fase 1 espera (ele seleciona colunas
  **por nome**, não por posição) — não é preciso montar manualmente as 17
  variáveis derivadas; o pipeline faz isso internamente.
- `predict_proba(...)[0][1]` pega a probabilidade da **classe positiva**
  (índice 1 = `stroke = 1`), a mesma convenção usada durante o treino.
- A resposta usa `st.error` (vermelho) para risco identificado e `st.success`
  (verde) para sem risco, sempre acompanhada da probabilidade formatada em
  padrão brasileiro (`formatar_pct`, vírgula decimal) e de uma barra de
  progresso (`st.progress`) como reforço visual da magnitude do risco.
- Um `st.expander` opcional mostra `dados_paciente` em formato bruto
  (`st.json`), dando transparência sobre exatamente o que foi enviado ao
  modelo — útil tanto para depuração quanto para a apresentação do projeto.

## 4.6 Execução

```bash
streamlit run app.py
```

Deve ser executado a partir da raiz do repositório (ou com o `.joblib`
presente em `modelos/` relativo ao próprio `app.py`), após rodar
`python treino_modelo.py` pelo menos uma vez.

## 4.7 Nota de ambiente: por que o `requirements.txt` fixa `pandas<3`

Durante o desenvolvimento, constatou-se que o **pandas 3.0** (lançado com um
novo backend de string via PyArrow) provoca uma falha nativa (segmentation
fault) na thread de execução de scripts do Streamlit neste ambiente — tanto
ao construir o `DataFrame` de uma linha quanto, mais adiante, ao chamar
`st.dataframe`/`st.table` (que serializam o DataFrame para Arrow
internamente). O problema não aparece em uma thread Python pura, apenas no
caminho específico Streamlit + Arrow.

Duas mitigações foram aplicadas e verificadas (o app foi testado de ponta a
ponta com `streamlit.testing.v1.AppTest`, incluindo os cenários de paciente de
alto e de baixo risco, sem falhas):

1. `requirements.txt` fixa `pandas>=2.0,<3.0` — a mesma série de pandas
   distribuída no Google Colab, então o comportamento é idêntico entre
   ambiente local e Colab.
2. A exibição dos dados enviados ao modelo usa `st.json(dados_paciente)`
   (um dicionário Python simples) em vez de `st.dataframe`, evitando por
   completo o caminho de serialização Arrow que causava o crash.
