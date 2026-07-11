# %% [markdown]
# # Sistema Inteligente de Classificação — Previsão de AVC
#
# **Problema:** prever o risco de AVC (Acidente Vascular Cerebral) a partir de dados
# clínicos e demográficos de pacientes — uma **classificação binária desbalanceada**
# (a classe positiva, "teve AVC", representa apenas ~5% dos registros).
#
# **Dataset:** Stroke Prediction Dataset (Kaggle) — `healthcare-dataset-stroke-data.csv`.
#
# **Algoritmos:** KNN (K-Nearest Neighbors) e Árvore de Decisão.
#
# **Métrica decisória:** Recall da classe AVC. Em contexto clínico, o erro mais grave
# é o **falso negativo** (não detectar um paciente que terá AVC); o Recall mede
# exatamente a capacidade de "não deixar escapar" os casos positivos.
#
# **Estrutura deste script** (cada seção corresponde a uma fase do projeto):
# 1. Carregamento e análise exploratória breve
# 2. Fase 1 — Pré-processamento (limpeza + transformações via pipeline)
# 3. Split estratificado 80/20 + SMOTE (somente no treino, via pipeline do imbalanced-learn)
# 4. Fase 2 — Modelagem com laço manual de hiperparâmetros
# 5. Fase 3 — Avaliação: matrizes de confusão (heatmaps) + tabela comparativa
# 6. Análise complementar — efeito do SMOTE e interpretabilidade da Árvore
# 7. Seleção do melhor modelo e salvamento do pipeline completo em `.joblib`
#
# > **Separação de conceitos:** este arquivo contém TODO o treino. A interface
# > (`app.py`) apenas carrega o `.joblib` salvo aqui — nenhuma lógica é duplicada.

# %%
# === Instalação de dependências ===
# No Google Colab, o 'imbalanced-learn' costuma vir pré-instalado; se necessário,
# descomente a linha abaixo (as demais bibliotecas são padrão do Colab):
# %pip install -q imbalanced-learn

# %%
# === Imports e configurações globais ===
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline as PipelineSklearn
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.tree import DecisionTreeClassifier

# O Pipeline do imbalanced-learn (e não o do scikit-learn) é obrigatório aqui:
# ele aplica o SMOTE APENAS durante o fit (treino) e o ignora na predição,
# o que evita vazamento de dados e permite salvar tudo em um único artefato.
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as PipelineImblearn
except ImportError as exc:  # mensagem amigável caso a dependência falte
    raise SystemExit(
        "Biblioteca 'imbalanced-learn' não encontrada. "
        "Instale com: pip install imbalanced-learn"
    ) from exc

# Semente fixa em TODAS as etapas estocásticas (split, SMOTE, Árvore) para que
# qualquer pessoa que rode este script obtenha exatamente os mesmos números.
RANDOM_STATE = 42

# Caminho parametrizável do CSV. No Colab, faça o upload manual descomentando:
#   from google.colab import files
#   files.upload()
# e o arquivo ficará disponível no diretório corrente ("healthcare-dataset-stroke-data.csv").
CANDIDATOS_CSV = [
    "data/healthcare-dataset-stroke-data.csv",  # estrutura deste repositório
    "healthcare-dataset-stroke-data.csv",       # upload direto no Colab
]

# Pastas de saída dos artefatos (criadas automaticamente se não existirem):
# modelos/    -> pipeline treinado (.joblib), consumido pela interface (app.py)
# resultados/ -> figuras e tabelas de métricas, usadas no relatório
PASTA_MODELOS = Path("modelos")
PASTA_RESULTADOS = Path("resultados")
PASTA_MODELOS.mkdir(exist_ok=True)
PASTA_RESULTADOS.mkdir(exist_ok=True)
CAMINHO_MODELO = PASTA_MODELOS / "modelo_avc.joblib"

# Paleta e "tinta" dos gráficos (tons validados para acessibilidade/daltonismo):
# heatmaps usam UMA única cor (azul) do claro ao escuro — cor sequencial codifica
# magnitude; jamais usar arco-íris em matriz de confusão.
RAMPA_AZUL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
COR_BARRA_CLARA, COR_BARRA_ESCURA = "#86b6ef", "#1c5cab"
TINTA_PRIMARIA, TINTA_SECUNDARIA, TINTA_SUAVE = "#0b0b0b", "#52514e", "#898781"
COR_GRADE, COR_EIXO = "#e1e0d9", "#c3c2b7"

plt.rcParams.update({
    "figure.facecolor": "#ffffff",
    "axes.facecolor": "#ffffff",
    "axes.edgecolor": COR_EIXO,
    "axes.labelcolor": TINTA_PRIMARIA,
    "text.color": TINTA_PRIMARIA,
    "xtick.color": TINTA_SUAVE,
    "ytick.color": TINTA_SUAVE,
    "grid.color": COR_GRADE,
    "font.family": "sans-serif",
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
})

pd.set_option("display.width", 130)
pd.set_option("display.max_columns", 30)


def formatar_pct(valor: float) -> str:
    """Formata proporção como percentual em padrão brasileiro (vírgula decimal)."""
    return f"{valor * 100:.1f}".replace(".", ",") + "%"


# %% [markdown]
# ## 1. Carregamento e análise exploratória breve
#
# Objetivos desta seção: conhecer os tipos das colunas, **medir o desbalanceamento
# do alvo**, contar os nulos reais de `bmi` e localizar particularidades
# (registro com `gender = "Other"`, duplicatas).

# %%
def carregar_dados(candidatos: list[str]) -> pd.DataFrame:
    """Carrega o CSV a partir do primeiro caminho existente na lista.

    O pandas já converte a string "N/A" em NaN por padrão, mas deixamos o
    parâmetro explícito para documentar que os nulos de `bmi` vêm nesse formato.
    """
    for caminho in candidatos:
        if Path(caminho).exists():
            print(f"Lendo dataset de: {caminho}")
            return pd.read_csv(caminho, na_values=["N/A"])
    raise FileNotFoundError(
        f"CSV não encontrado. Coloque o arquivo em um destes caminhos: {candidatos}"
    )


df = carregar_dados(CANDIDATOS_CSV)
print(f"\nDimensões: {df.shape[0]} linhas x {df.shape[1]} colunas")
df.head()

# %%
# Tipos das colunas e contagem de não-nulos — repare que `bmi` é a única com nulos.
df.info()

# %%
# --- Distribuição da variável-alvo: o coração do problema ---
contagem_alvo = df["stroke"].value_counts().sort_index()
proporcao_alvo = df["stroke"].value_counts(normalize=True).sort_index()
print("Distribuição do alvo (stroke):")
print(f"  0 (sem AVC): {contagem_alvo[0]} pacientes ({formatar_pct(proporcao_alvo[0])})")
print(f"  1 (com AVC): {contagem_alvo[1]} pacientes ({formatar_pct(proporcao_alvo[1])})")

# Nulos por coluna — NÃO assumimos um valor fixo; medimos no dado carregado.
print("\nValores nulos por coluna:")
print(df.isna().sum()[df.isna().sum() > 0].to_string())

# Particularidades que precisaremos tratar:
print(f"\nRegistros duplicados (linhas idênticas): {df.duplicated().sum()}")
print(f"Contagem de 'gender':\n{df['gender'].value_counts().to_string()}")

# %%
# Estatísticas descritivas das numéricas contínuas (escalas MUITO diferentes:
# idade vai até ~82, glicose até ~272 — por isso o MinMaxScaler será essencial
# para o KNN, que compara pacientes por distância).
df[["age", "avg_glucose_level", "bmi"]].describe().round(2)

# %%
# Gráfico da distribuição do alvo — evidencia visualmente o desbalanceamento
# que justifica o SMOTE e a escolha do Recall como métrica decisória.
fig, ax = plt.subplots(figsize=(6.4, 4.2))
barras = ax.bar(
    ["Sem AVC (0)", "Com AVC (1)"],
    contagem_alvo.values,
    color=[COR_BARRA_CLARA, COR_BARRA_ESCURA],
    width=0.55,
)
for barra, qtd, prop in zip(barras, contagem_alvo.values, proporcao_alvo.values):
    ax.annotate(
        f"{qtd}\n({formatar_pct(prop)})",
        xy=(barra.get_x() + barra.get_width() / 2, barra.get_height()),
        xytext=(0, 5), textcoords="offset points",
        ha="center", va="bottom", fontsize=11, color=TINTA_PRIMARIA,
    )
ax.set_title("Distribuição da variável-alvo (stroke)", loc="left",
             fontsize=13, fontweight="bold", color=TINTA_PRIMARIA, pad=32)
ax.text(0, 1.03, "Apenas ~5% dos pacientes tiveram AVC — classificação fortemente desbalanceada",
        transform=ax.transAxes, fontsize=9.5, color=TINTA_SECUNDARIA)
ax.set_ylabel("Número de pacientes")
ax.set_ylim(0, contagem_alvo.max() * 1.18)
ax.grid(axis="y", linewidth=0.6)
ax.set_axisbelow(True)
for lado in ("top", "right"):
    ax.spines[lado].set_visible(False)
fig.tight_layout()
fig.savefig(PASTA_RESULTADOS / "fig_distribuicao_alvo.png")
plt.show()

# %% [markdown]
# ## 2. Fase 1 — Pré-processamento
#
# Dividimos o pré-processamento em dois grupos, e essa distinção é um ponto
# central do projeto:
#
# 1. **Limpeza estrutural** (feita aqui, antes do split): operações que apenas
#    removem linhas/colunas e **não aprendem nada dos dados** — remover `id`,
#    duplicatas e o registro único com `gender = "Other"`.
# 2. **Transformações que aprendem parâmetros** (imputação da mediana, One-Hot,
#    MinMax): entram **dentro do pipeline**, para que sejam ajustadas **somente
#    com o conjunto de treino** (evitando vazamento de dados) e reaplicadas
#    automaticamente a qualquer dado novo — inclusive na interface Streamlit,
#    que envia dados brutos ao pipeline salvo.

# %%
# --- 2.1 Limpeza estrutural ---
df_limpo = df.drop(columns=["id"])  # identificador não tem valor preditivo

qtd_duplicatas = df_limpo.duplicated().sum()
df_limpo = df_limpo.drop_duplicates()
print(f"Duplicatas removidas: {qtd_duplicatas}")

# Tratamento do 'gender = Other': existe apenas 1 registro no dataset inteiro.
# Uma única instância não permite ao modelo aprender um padrão para a categoria
# e quebraria a codificação binária de `gender`; a opção mais simples e honesta
# é removê-la, documentando a decisão (perda de 0,02% dos dados).
qtd_other = (df_limpo["gender"] == "Other").sum()
df_limpo = df_limpo[df_limpo["gender"] != "Other"].reset_index(drop=True)
print(f"Registros com gender='Other' removidos: {qtd_other}")
print(f"Dimensões após a limpeza: {df_limpo.shape[0]} linhas x {df_limpo.shape[1]} colunas")

# %%
# --- 2.2 Definição das transformações por grupo de colunas ---
# Numéricas contínuas: imputação da MEDIANA (robusta a outliers — a média seria
# puxada pelos valores extremos de bmi/glicose) + normalização 0-1 (MinMaxScaler).
# A escala é crítica para o KNN: sem ela, a glicose (até ~272) dominaria a
# distância e a idade/bmi quase não contariam.
COLUNAS_NUMERICAS = ["age", "avg_glucose_level", "bmi"]

# Binárias (2 categorias): viram UMA coluna 0/1 — equivalente ao Label Encoding.
# Implementamos com OneHotEncoder(drop='if_binary') para que o mapeamento
# aprendido no treino fique guardado dentro do pipeline.
COLUNAS_BINARIAS = ["gender", "ever_married", "Residence_type"]

# Nominais sem ordem (>2 categorias): One-Hot Encoding — usar Label Encoding aqui
# criaria uma ordem artificial (ex.: Private=3 > Govt_job=1) que os modelos
# interpretariam como magnitude, distorcendo distâncias e divisões da árvore.
COLUNAS_NOMINAIS = ["work_type", "smoking_status"]

# Já são 0/1 no dataset original — passam direto, sem transformação.
COLUNAS_JA_BINARIAS = ["hypertension", "heart_disease"]


def construir_preprocessador() -> ColumnTransformer:
    """Monta o ColumnTransformer com uma rota de transformação por tipo de coluna."""
    rota_numericas = PipelineSklearn(steps=[
        ("imputador_mediana", SimpleImputer(strategy="median")),
        ("normalizador", MinMaxScaler()),
    ])
    return ColumnTransformer(
        transformers=[
            ("num", rota_numericas, COLUNAS_NUMERICAS),
            ("bin", OneHotEncoder(drop="if_binary", sparse_output=False), COLUNAS_BINARIAS),
            # handle_unknown='ignore': se surgir uma categoria nova em produção,
            # a aplicação não quebra (a linha vira zeros nas colunas one-hot).
            ("nom", OneHotEncoder(handle_unknown="ignore", sparse_output=False), COLUNAS_NOMINAIS),
            ("direto", "passthrough", COLUNAS_JA_BINARIAS),
        ],
        remainder="drop",  # qualquer coluna extra é descartada explicitamente
    )


# %% [markdown]
# ## 3. Split estratificado + SMOTE via pipeline
#
# - **80/20 com `stratify=y`**: o teste preserva a proporção real de ~5% de AVC —
#   sem estratificar, a classe rara poderia ficar sub-representada no teste.
# - **SMOTE apenas no treino, DEPOIS do split**: balancear antes do split (ou no
#   conjunto todo) faria exemplos sintéticos "vazarem" para o teste, inflando as
#   métricas. O `Pipeline` do imbalanced-learn garante isso por construção:
#   o passo SMOTE só é executado no `fit`, nunca no `predict`.

# %%
X = df_limpo.drop(columns=["stroke"])
y = df_limpo["stroke"]

X_treino, X_teste, y_treino, y_teste = train_test_split(
    X, y, test_size=0.20, stratify=y, random_state=RANDOM_STATE
)

print(f"Treino: {X_treino.shape[0]} pacientes | Teste: {X_teste.shape[0]} pacientes")
print(f"Proporção de AVC no treino: {formatar_pct(y_treino.mean())}")
print(f"Proporção de AVC no teste : {formatar_pct(y_teste.mean())}")

# %%
# --- Demonstração didática do efeito do SMOTE (APENAS ilustrativa) ---
# O fluxo oficial de treino usa o pipeline; este bloco só materializa o "antes e
# depois" do balanceamento para visualizarmos o que o SMOTE faz com o treino:
# ele cria exemplos SINTÉTICOS da classe minoritária interpolando vizinhos
# próximos (por isso é aplicado após a normalização, no espaço transformado).
preprocessador_demo = construir_preprocessador()
X_treino_transformado = preprocessador_demo.fit_transform(X_treino)
_, y_apos_smote = SMOTE(random_state=RANDOM_STATE).fit_resample(
    X_treino_transformado, y_treino
)
print("Classe 0 / Classe 1 no treino ANTES do SMOTE :",
      (y_treino == 0).sum(), "/", (y_treino == 1).sum())
print("Classe 0 / Classe 1 no treino DEPOIS do SMOTE:",
      (y_apos_smote == 0).sum(), "/", (y_apos_smote == 1).sum())
print("O conjunto de TESTE permanece intocado, com a proporção real de ~5%.")

# %% [markdown]
# ## 4. Fase 2 — Modelagem com laço manual de hiperparâmetros
#
# Exploramos os hiperparâmetros com **laços `for` explícitos** (decisão didática:
# cada combinação fica visível e explicável, sem a "caixa-preta" do GridSearchCV):
#
# - **KNN**: K ∈ {3, 5, 7} × distância ∈ {Euclidiana, Manhattan} → 6 combinações;
# - **Árvore de Decisão**: `max_depth` ∈ {3, 5, 10, None} → 4 configurações,
#   para observar o efeito da profundidade sobre o overfitting.
#
# Como o balanceamento já é feito pelo SMOTE, **não** usamos `class_weight`
# na Árvore (as duas técnicas se sobreporiam e distorceriam a comparação).

# %%
def construir_pipeline(modelo) -> PipelineImblearn:
    """Pipeline completo: pré-processamento -> SMOTE (só no fit) -> classificador.

    É ESTE objeto único que será salvo em .joblib: quem o carrega envia dados
    brutos e recebe a predição, com a garantia de que as transformações são
    exatamente as aprendidas no treino.
    """
    return PipelineImblearn(steps=[
        ("preprocessamento", construir_preprocessador()),
        ("smote", SMOTE(random_state=RANDOM_STATE)),
        ("modelo", modelo),
    ])


def avaliar_modelo(nome: str, pipeline: PipelineImblearn) -> dict:
    """Treina o pipeline e calcula as métricas da Fase 3 sobre o conjunto de teste.

    Todas as métricas de classe usam pos_label=1 (AVC), pois é o desempenho na
    classe rara que interessa clinicamente.
    """
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


resultados = []

# --- Laço manual do KNN: 3 valores de K x 2 métricas de distância ---
NOME_DISTANCIA = {"euclidean": "Euclidiana", "manhattan": "Manhattan"}
for metrica_distancia in ["euclidean", "manhattan"]:
    for k in [3, 5, 7]:
        nome = f"KNN (K={k}, {NOME_DISTANCIA[metrica_distancia]})"
        knn = KNeighborsClassifier(n_neighbors=k, metric=metrica_distancia)
        resultados.append(avaliar_modelo(nome, construir_pipeline(knn)))
        print(f"Treinado: {nome}")

# --- Laço manual da Árvore de Decisão: 4 profundidades máximas ---
for profundidade in [3, 5, 10, None]:
    rotulo = profundidade if profundidade is not None else "None (sem limite)"
    nome = f"Árvore (max_depth={rotulo})"
    arvore = DecisionTreeClassifier(max_depth=profundidade, random_state=RANDOM_STATE)
    resultados.append(avaliar_modelo(nome, construir_pipeline(arvore)))
    print(f"Treinado: {nome}")

# %% [markdown]
# ## 5. Fase 3 — Avaliação
#
# ### 5.1 Por que a acurácia engana aqui
# Antes da tabela, medimos um **baseline trivial**: um "modelo" que prevê
# "sem AVC" para TODOS os pacientes. Ele não detecta nenhum caso da doença
# (Recall = 0) e ainda assim atinge ~95% de acurácia, porque acerta toda a
# classe majoritária. Conclusão: **em dados desbalanceados, acurácia alta não
# significa modelo útil** — por isso a decisão é ancorada no Recall da classe AVC.

# %%
# Baseline trivial calculado à mão (transparente): acerta apenas os negativos.
acuracia_baseline = (y_teste == 0).mean()
resultados.insert(0, {
    "Modelo": 'Baseline trivial (prevê "sem AVC" p/ todos)',
    "Acurácia": acuracia_baseline,
    "Precisão (AVC)": 0.0,   # nunca prevê a classe positiva
    "Recall (AVC)": 0.0,     # não detecta nenhum AVC -> clinicamente inútil
    "F1-Score (AVC)": 0.0,
    "matriz_confusao": None,
    "pipeline": None,
})
print(f"Acurácia do baseline trivial: {formatar_pct(acuracia_baseline)} — com Recall 0!")

# %%
# --- 5.2 Tabela comparativa consolidada (todas as combinações lado a lado) ---
COLUNAS_METRICAS = ["Modelo", "Acurácia", "Precisão (AVC)", "Recall (AVC)", "F1-Score (AVC)"]
tabela_resultados = (
    pd.DataFrame(resultados)[COLUNAS_METRICAS]
    .sort_values("Recall (AVC)", ascending=False)
    .reset_index(drop=True)
)
tabela_resultados.to_csv(PASTA_RESULTADOS / "resultados_comparativos.csv", index=False)
print("Tabela comparativa (ordenada pela métrica decisória, Recall da classe AVC):\n")
print(tabela_resultados.round(4).to_string())

# %%
# --- 5.3 Matrizes de confusão como heatmaps ---
# Convenção do scikit-learn: linhas = classe real, colunas = classe prevista.
#   [ [VN, FP],     VN = verdadeiro negativo | FP = falso positivo (alarme falso)
#     [FN, VP] ]    FN = falso NEGATIVO (o erro grave!) | VP = verdadeiro positivo
# A COR de cada célula é a proporção dentro da classe real (linha normalizada):
# assim a diagonal escura mostra, de uma vez, o Recall de cada classe — sem isso,
# a classe majoritária dominaria a escala de cor e os 249 casos de AVC sumiriam.
CMAP_AZUL = plt.matplotlib.colors.LinearSegmentedColormap.from_list("azul_seq", RAMPA_AZUL)
ROTULOS_CLASSES = ["Sem AVC", "AVC"]


def desenhar_matriz(ax, matriz: np.ndarray, titulo: str) -> None:
    """Desenha uma matriz de confusão anotada com contagem e % da classe real."""
    proporcao_linha = matriz / matriz.sum(axis=1, keepdims=True)
    anotacoes = np.array([
        [f"{matriz[i, j]}\n({formatar_pct(proporcao_linha[i, j])})" for j in range(2)]
        for i in range(2)
    ])
    sns.heatmap(
        proporcao_linha, ax=ax, cmap=CMAP_AZUL, vmin=0, vmax=1,
        annot=anotacoes, fmt="", annot_kws={"fontsize": 10},
        cbar=False, square=True, linewidths=2, linecolor="#ffffff",
        xticklabels=ROTULOS_CLASSES, yticklabels=ROTULOS_CLASSES,
    )
    ax.set_title(titulo, fontsize=10.5, color=TINTA_PRIMARIA, pad=8)
    ax.tick_params(length=0)


def plotar_grade_matrizes(lista_resultados: list[dict], titulo: str,
                          n_linhas: int, n_colunas: int, arquivo: Path) -> None:
    """Monta uma grade de heatmaps de matrizes de confusão e salva em PNG."""
    fig, eixos = plt.subplots(
        n_linhas, n_colunas, figsize=(3.3 * n_colunas + 1, 3.7 * n_linhas + 1)
    )
    for ax, res in zip(eixos.flat, lista_resultados):
        recall = formatar_pct(res["Recall (AVC)"])
        desenhar_matriz(ax, res["matriz_confusao"], f"{res['Modelo']}\nRecall (AVC) = {recall}")
    for ax in eixos.flat[len(lista_resultados):]:  # esconde eixos sobrando na grade
        ax.set_visible(False)
    # Rótulos de eixo apenas nas bordas da grade (evita repetição e sobreposição)
    for i, ax in enumerate(eixos.flat):
        na_primeira_coluna = i % n_colunas == 0
        na_ultima_linha = i // n_colunas == n_linhas - 1
        ax.set_ylabel("Classe real" if na_primeira_coluna else "")
        ax.set_xlabel("Classe prevista" if na_ultima_linha else "")
        if not na_primeira_coluna:
            ax.set_yticklabels([])
        if not na_ultima_linha:
            ax.set_xticklabels([])
    fig.suptitle(titulo, fontsize=13, fontweight="bold", color=TINTA_PRIMARIA, x=0.02, ha="left")
    fig.text(0.02, 0.945, "Cor = proporção dentro da classe real (linha); célula inferior esquerda = falsos negativos, o erro clínico grave",
             fontsize=9, color=TINTA_SECUNDARIA)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.subplots_adjust(hspace=0.30)  # respiro entre as linhas da grade
    fig.savefig(arquivo)
    plt.show()


resultados_knn = [r for r in resultados if r["Modelo"].startswith("KNN")]
resultados_arvore = [r for r in resultados if r["Modelo"].startswith("Árvore")]

plotar_grade_matrizes(resultados_knn, "Matrizes de confusão — KNN (teste)", 2, 3,
                      PASTA_RESULTADOS / "fig_matrizes_knn.png")
plotar_grade_matrizes(resultados_arvore, "Matrizes de confusão — Árvore de Decisão (teste)", 2, 2,
                      PASTA_RESULTADOS / "fig_matrizes_arvore.png")

# %% [markdown]
# ## 6. Análises complementares
#
# ### 6.1 Efeito do SMOTE (ablação)
# Para comprovar que o SMOTE é quem viabiliza a detecção da classe rara,
# retreinamos a melhor configuração de cada algoritmo **sem** o passo SMOTE
# (mesmo pré-processamento, mesmo split) e comparamos as métricas.

# %%
melhor_knn = max(resultados_knn, key=lambda r: (r["Recall (AVC)"], r["F1-Score (AVC)"]))
melhor_arvore = max(resultados_arvore, key=lambda r: (r["Recall (AVC)"], r["F1-Score (AVC)"]))

comparacao_smote = []
for res_com_smote in [melhor_knn, melhor_arvore]:
    # clona a configuração do classificador vencedor e monta o pipeline SEM SMOTE
    modelo_base = res_com_smote["pipeline"].named_steps["modelo"]
    pipeline_sem_smote = PipelineSklearn(steps=[
        ("preprocessamento", construir_preprocessador()),
        ("modelo", modelo_base.__class__(**modelo_base.get_params())),
    ])
    pipeline_sem_smote.fit(X_treino, y_treino)
    y_previsto = pipeline_sem_smote.predict(X_teste)
    for cenario, recall, precisao, acuracia in [
        ("com SMOTE", res_com_smote["Recall (AVC)"], res_com_smote["Precisão (AVC)"],
         res_com_smote["Acurácia"]),
        ("SEM SMOTE", recall_score(y_teste, y_previsto, zero_division=0),
         precision_score(y_teste, y_previsto, zero_division=0),
         accuracy_score(y_teste, y_previsto)),
    ]:
        comparacao_smote.append({
            "Modelo": res_com_smote["Modelo"], "Cenário": cenario,
            "Recall (AVC)": recall, "Precisão (AVC)": precisao, "Acurácia": acuracia,
        })

tabela_smote = pd.DataFrame(comparacao_smote)
tabela_smote.to_csv(PASTA_RESULTADOS / "resultados_efeito_smote.csv", index=False)
print("Efeito do SMOTE nas melhores configurações (conjunto de teste):\n")
print(tabela_smote.round(4).to_string(index=False))

# %% [markdown]
# ### 6.2 Interpretabilidade — importância das variáveis na melhor Árvore
# Diferentemente do KNN (que não explica suas decisões), a Árvore expõe quanto
# cada variável contribuiu para reduzir a impureza das divisões — um ganho de
# interpretabilidade relevante em contexto clínico.

# %%
preprocessador_ajustado = melhor_arvore["pipeline"].named_steps["preprocessamento"]
nomes_variaveis = [
    nome.split("__", 1)[1]  # remove o prefixo da rota (num__, bin__, nom__, direto__)
    for nome in preprocessador_ajustado.get_feature_names_out()
]
importancias = pd.Series(
    melhor_arvore["pipeline"].named_steps["modelo"].feature_importances_,
    index=nomes_variaveis,
).sort_values(ascending=False)
print(f"Top 10 variáveis mais importantes — {melhor_arvore['Modelo']}:\n")
print(importancias.head(10).round(4).to_string())

# %% [markdown]
# ## 7. Seleção do melhor modelo e salvamento
#
# **Critério de decisão:** maior **Recall da classe AVC** no teste (empates são
# resolvidos pelo F1-Score, que equilibra a precisão). A justificativa é clínica:
# um falso negativo significa mandar para casa, sem acompanhamento, um paciente
# que terá AVC — erro muito mais grave que um alarme falso, que apenas gera
# exames adicionais.

# %%
candidatos = [r for r in resultados if r["pipeline"] is not None]
melhor = max(candidatos, key=lambda r: (r["Recall (AVC)"], r["F1-Score (AVC)"]))

print("=" * 72)
print(f"MELHOR MODELO: {melhor['Modelo']}")
print(f"  Recall (AVC) = {formatar_pct(melhor['Recall (AVC)'])}  <- métrica decisória")
print(f"  F1-Score (AVC) = {melhor['F1-Score (AVC)']:.4f} | "
      f"Precisão (AVC) = {melhor['Precisão (AVC)']:.4f} | "
      f"Acurácia = {melhor['Acurácia']:.4f}")
print("=" * 72)

# Relatório completo por classe do campeão (visão detalhada das métricas):
y_previsto_melhor = melhor["pipeline"].predict(X_teste)
print("\nClassification report do melhor modelo (conjunto de teste):\n")
print(classification_report(y_teste, y_previsto_melhor,
                            target_names=["Sem AVC (0)", "AVC (1)"], zero_division=0))

# %%
# --- Salvamento do PIPELINE COMPLETO (pré-processamento + SMOTE + modelo) ---
# Salvamos o pipeline treinado no conjunto de treino — exatamente o objeto que
# gerou as métricas reportadas acima, garantindo total rastreabilidade entre o
# relatório e o artefato entregue à interface.
joblib.dump(melhor["pipeline"], CAMINHO_MODELO)
print(f"Pipeline salvo em: {CAMINHO_MODELO}")

# %%
# --- Prova final: simulação do fluxo da interface ---
# Recarregamos o .joblib e enviamos um paciente FICTÍCIO com dados BRUTOS
# (mesmo formato que o formulário do app envia). Nenhuma transformação manual:
# o pipeline aplica imputação, encoding e escala sozinho.
pipeline_carregado = joblib.load(CAMINHO_MODELO)
paciente_exemplo = pd.DataFrame([{
    "gender": "Male",
    "age": 67.0,
    "hypertension": 1,
    "heart_disease": 1,
    "ever_married": "Yes",
    "work_type": "Private",
    "Residence_type": "Urban",
    "avg_glucose_level": 228.69,
    "bmi": 36.6,
    "smoking_status": "formerly smoked",
}])
predicao = pipeline_carregado.predict(paciente_exemplo)[0]
probabilidade_avc = pipeline_carregado.predict_proba(paciente_exemplo)[0][1]
print("Paciente fictício de alto risco ->",
      "COM RISCO DE AVC" if predicao == 1 else "SEM RISCO DE AVC",
      f"(probabilidade de AVC: {formatar_pct(probabilidade_avc)})")
print("\nTreino concluído. Artefatos gerados:")
print(f"  {CAMINHO_MODELO}  (pipeline completo, consumido pelo app.py)")
print(f"  {PASTA_RESULTADOS}/  (tabelas de métricas em .csv e figuras fig_*.png)")
