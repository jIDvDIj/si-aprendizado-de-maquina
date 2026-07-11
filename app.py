"""Aplicação web (Streamlit) — Preditor de Risco de AVC.

SEPARAÇÃO DE CONCEITOS (critério central do projeto):
este arquivo NÃO contém nenhuma lógica de treino, pré-processamento manual
ou métrica. Ele apenas:

1. carrega o PIPELINE COMPLETO salvo por ``treino_modelo.py`` (modelo_avc.joblib),
   que já embute imputação de nulos, encoding, normalização e o classificador;
2. coleta os dados BRUTOS do paciente em um formulário;
3. envia esses dados ao pipeline e exibe a predição com a probabilidade.

Como o pipeline reaplica sozinho as transformações aprendidas no treino,
não há risco de a interface transformar os dados de um jeito diferente
(fonte clássica de bugs quando treino e aplicação são acoplados).

Execução:  streamlit run app.py
"""

from pathlib import Path

import joblib
import pandas as pd
import streamlit as st

# O caminho é resolvido em relação a ESTE arquivo, e não ao diretório corrente,
# para que o app funcione independentemente de onde o comando for executado.
CAMINHO_MODELO = Path(__file__).parent / "modelos" / "modelo_avc.joblib"

# ---------------------------------------------------------------------------
# Dicionários de tradução: o formulário fala português com o usuário, mas o
# pipeline espera as categorias ORIGINAIS do dataset (em inglês). A conversão
# acontece aqui, na borda da interface — o modelo nunca vê rótulos traduzidos.
# ---------------------------------------------------------------------------
OPCOES_GENERO = {"Feminino": "Female", "Masculino": "Male"}
OPCOES_SIM_NAO = {"Não": 0, "Sim": 1}
OPCOES_CASADO = {"Não": "No", "Sim": "Yes"}
OPCOES_TRABALHO = {
    "Setor privado": "Private",
    "Autônomo(a)": "Self-employed",
    "Funcionário(a) público(a)": "Govt_job",
    "Criança": "children",
    "Nunca trabalhou": "Never_worked",
}
OPCOES_RESIDENCIA = {"Urbana": "Urban", "Rural": "Rural"}
OPCOES_FUMANTE = {
    "Nunca fumou": "never smoked",
    "Fumou no passado": "formerly smoked",
    "Fuma atualmente": "smokes",
    "Não informado": "Unknown",
}


@st.cache_resource
def carregar_pipeline(caminho: Path):
    """Carrega o pipeline uma única vez e o mantém em cache entre interações."""
    return joblib.load(caminho)


def formatar_pct(valor: float) -> str:
    """Percentual em padrão brasileiro (vírgula decimal)."""
    return f"{valor * 100:.1f}".replace(".", ",") + "%"


st.set_page_config(page_title="Preditor de Risco de AVC", page_icon="🧠", layout="centered")

st.title("🧠 Preditor de Risco de AVC")
st.markdown(
    "Preencha os dados do paciente e receba a **estimativa de risco de AVC** "
    "calculada pelo modelo de machine learning treinado no projeto."
)

# Falha amigável caso o artefato de treino ainda não exista.
if not CAMINHO_MODELO.exists():
    st.error(
        "Arquivo `modelos/modelo_avc.joblib` não encontrado. "
        "Execute primeiro o treino: `python treino_modelo.py`."
    )
    st.stop()

pipeline = carregar_pipeline(CAMINHO_MODELO)
nome_classificador = type(pipeline.named_steps["modelo"]).__name__
st.caption(
    f"Modelo em uso: **{nome_classificador}** (pipeline completo com "
    "pré-processamento embutido, selecionado pelo maior Recall da classe AVC)."
)

# ---------------------------------------------------------------------------
# Formulário: os campos reproduzem exatamente as variáveis brutas do dataset.
# st.form evita reprocessar a página a cada campo alterado — a predição só
# roda quando o usuário clica em "Calcular risco".
# ---------------------------------------------------------------------------
with st.form("formulario_paciente"):
    coluna_esquerda, coluna_direita = st.columns(2)

    with coluna_esquerda:
        idade = st.number_input("Idade (anos)", min_value=1, max_value=110,
                                value=45, step=1, key="idade")
        genero = st.selectbox("Gênero", list(OPCOES_GENERO), key="genero")
        casado = st.selectbox("Já foi casado(a)?", list(OPCOES_CASADO), key="casado")
        trabalho = st.selectbox("Tipo de trabalho", list(OPCOES_TRABALHO), key="trabalho")
        residencia = st.selectbox("Tipo de residência", list(OPCOES_RESIDENCIA), key="residencia")

    with coluna_direita:
        glicose = st.number_input("Nível médio de glicose (mg/dL)", min_value=40.0,
                                  max_value=300.0, value=100.0, step=1.0, key="glicose")
        imc = st.number_input("IMC — Índice de Massa Corporal", min_value=10.0,
                              max_value=70.0, value=25.0, step=0.1, key="imc")
        fumante = st.selectbox("Status de fumante", list(OPCOES_FUMANTE), key="fumante")
        hipertensao = st.selectbox("Tem hipertensão?", list(OPCOES_SIM_NAO), key="hipertensao")
        cardiaco = st.selectbox("Tem doença cardíaca?", list(OPCOES_SIM_NAO), key="cardiaco")

    enviado = st.form_submit_button("🩺 Calcular risco", use_container_width=True)

if enviado:
    # Dicionário com as colunas brutas EXATAS que o pipeline espera
    # (mesmos nomes do CSV original — o ColumnTransformer seleciona por nome).
    dados_paciente = {
        "gender": OPCOES_GENERO[genero],
        "age": float(idade),
        "hypertension": OPCOES_SIM_NAO[hipertensao],
        "heart_disease": OPCOES_SIM_NAO[cardiaco],
        "ever_married": OPCOES_CASADO[casado],
        "work_type": OPCOES_TRABALHO[trabalho],
        "Residence_type": OPCOES_RESIDENCIA[residencia],
        "avg_glucose_level": float(glicose),
        "bmi": float(imc),
        "smoking_status": OPCOES_FUMANTE[fumante],
    }
    paciente = pd.DataFrame([dados_paciente])  # DataFrame de 1 linha, dados brutos

    predicao = int(pipeline.predict(paciente)[0])
    # predict_proba: coluna 1 = probabilidade da classe positiva (AVC).
    probabilidade_avc = float(pipeline.predict_proba(paciente)[0][1])

    st.divider()
    if predicao == 1:
        st.error(
            f"⚠️ **Risco de AVC identificado** — probabilidade estimada: "
            f"**{formatar_pct(probabilidade_avc)}**"
        )
        st.markdown("Recomenda-se **avaliação médica** para investigação detalhada.")
    else:
        st.success(
            f"✅ **Sem risco de AVC identificado** — probabilidade estimada: "
            f"**{formatar_pct(probabilidade_avc)}**"
        )

    st.progress(min(max(probabilidade_avc, 0.0), 1.0),
                text=f"Probabilidade de AVC: {formatar_pct(probabilidade_avc)}")

    # Transparência: mostra exatamente o que foi enviado ao modelo.
    with st.expander("Ver dados enviados ao modelo (formato bruto do dataset)"):
        st.json(dados_paciente)

st.divider()
st.caption(
    "⚕️ Projeto acadêmico de machine learning (ADS). Esta ferramenta **não** "
    "substitui diagnóstico ou aconselhamento médico profissional."
)
