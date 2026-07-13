# Sistema Inteligente de Classificação — Previsão de AVC

Projeto acadêmico (ADS) de **classificação binária desbalanceada**: prever o risco
de AVC a partir de dados clínicos e demográficos do *Stroke Prediction Dataset*
(Kaggle), comparando **KNN** e **Árvore de Decisão** com balanceamento por **SMOTE**.

## Estrutura do projeto

A organização das pastas reflete a **separação de conceitos** exigida pelo
projeto: o treino e a interface são arquivos independentes, ligados apenas pelo
artefato `modelos/modelo_avc.joblib`.

```
├── treino_modelo.py        # TODO o treino: EDA, pré-processamento, SMOTE via
│                           # pipeline, laço manual de hiperparâmetros, avaliação
│                           # e salvamento do pipeline vencedor
├── app.py                  # SÓ a interface (Streamlit): carrega o .joblib e
│                           # faz predições — nenhuma lógica de treino
├── requirements.txt        # dependências
├── data/
│   └── healthcare-dataset-stroke-data.csv   # dataset (Kaggle)
├── notebooks/
│   └── treino_modelo.ipynb # o treino em formato notebook (Google Colab)
├── modelos/                # [gerado pelo treino]
│   └── modelo_avc.joblib   # pipeline completo: pré-processamento + SMOTE + modelo
├── resultados/             # [gerado pelo treino]
│   ├── fig_distribuicao_alvo.png
│   ├── fig_matrizes_knn.png
│   ├── fig_matrizes_arvore.png
│   ├── resultados_comparativos.csv
│   └── resultados_efeito_smote.csv
└── docs/
    ├── relatorio.md                    # relatório crítico (metodologia, métricas, discussão)
    ├── fase1-preprocessamento.md       # doc. técnica: limpeza, encoding, escala (Fase 1)
    ├── fase2-modelagem.md              # doc. técnica: split, SMOTE, laço de hiperparâmetros (Fase 2)
    ├── fase3-avaliacao.md              # doc. técnica: métricas, heatmaps, seleção do modelo (Fase 3)
    ├── fase-bonus-deployment.md        # doc. técnica: interface Streamlit (Fase Bônus)
    ├── base-projeto.md                 # enunciado do projeto
    └── prompt-base.md                  # especificação técnica das decisões
```

## Documentação por fase

Cada fase do projeto tem um documento técnico dedicado, detalhando **o que foi
feito com os dados** e **como o código implementa cada decisão** (com números
reais extraídos do dataset e do pipeline treinado, não estimativas):

- [`docs/fase1-preprocessamento.md`](docs/fase1-preprocessamento.md) — análise
  exploratória, limpeza estrutural, imputação, encoding e normalização.
- [`docs/fase2-modelagem.md`](docs/fase2-modelagem.md) — split estratificado,
  balanceamento com SMOTE via pipeline, laço manual de hiperparâmetros.
- [`docs/fase3-avaliacao.md`](docs/fase3-avaliacao.md) — métricas, matrizes de
  confusão, ablação do SMOTE, seleção e salvamento do melhor modelo.
- [`docs/fase-bonus-deployment.md`](docs/fase-bonus-deployment.md) — interface
  Streamlit, separação treino/interface, nota de ambiente (pandas/Streamlit).

O [`docs/relatorio.md`](docs/relatorio.md) é a síntese executiva (o texto
pronto para o documento entregável); os documentos de fase acima são o
aprofundamento técnico de apoio à apresentação.

## Como executar

Sempre a partir da **raiz do projeto**:

```bash
pip install -r requirements.txt

# 1) Treino — gera modelos/modelo_avc.joblib e a pasta resultados/
python treino_modelo.py

# 2) Interface web
streamlit run app.py
```

No **Google Colab**: abra `notebooks/treino_modelo.ipynb`, faça o upload do CSV
(`files.upload()`) ou ajuste o caminho em `CANDIDATOS_CSV` e execute as células
em ordem (as pastas de saída são criadas automaticamente).

## Resultado principal

O melhor modelo pela métrica decisória (**Recall da classe AVC**) foi a
**Árvore de Decisão com `max_depth=5`**: detecta **82%** dos casos de AVC do
conjunto de teste (41 de 50), contra 40% do melhor KNN. Sem o SMOTE, o mesmo
modelo detectaria apenas 4%. A discussão completa está em `docs/relatorio.md`.
