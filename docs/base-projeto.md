Projeto Prático: Sistema Inteligente de Classificação 
1. Visão Geral do Projeto 
Desenvolver uma aplicação que resolva um problema de classificação real utilizando pelo menos dois dos três algoritmos clássicos estudados em sala (a escolha): KNN,  Árvores de Decisão e Naive Bayes. 
Sugestões de Temas 
• Churn de Clientes: Prever se um usuário vai cancelar a assinatura de um  serviço. 
• Diagnóstico de Saúde Baseado em Sintomas: Classificar potenciais condições  médicas. 
• Ou outro tema qualquer, a escolha do estudante. 
2. Escopo Técnico e Arquitetura do Projeto 
O projeto será dividido em 3 fases obrigatórias: 
Fase 1: Análise Exploratória e Pré-processamento  
• Limpeza de dados (tratamento de valores nulos e duplicados). • Conversão de variáveis categóricas (One-Hot Encoding / Label Encoding). • Normalização/Padronização dos dados (fundamental para o KNN). • Divisão do dataset em Treino e Teste (ex: 80/20). 
Fase 2: Modelagem e Treinamento 
Os alunos devem treinar os modelos utilizando bibliotecas padrão (como scikit-learn  em Python): 
1. KNN (K-Nearest Neighbors): Testar diferentes valores de K (ex: 3, 5, 7) e  métricas de distância (Euclidiana vs. Manhattan). 
2. Árvore de Decisão: Avaliar o impacto da profundidade máxima (max_depth)  para evitar overfitting. 
3. Naive Bayes: Aplicar a variante correta (ex: GaussianNB para dados contínuos  ou MultinomialNB para textos/contagens). 
Fase 3: Avaliação de Métricas 
Cada grupo deve gerar e comparar: 
• Matriz de Confusão para cada modelo.
• Métricas: Acurácia, Precisão, Recall e F1-Score. 
• Relatório Crítico: Um texto objetivo justificando qual modelo foi o melhor para  o cenário escolhido e o porquê. Apresentar tabelas/ gráficos com os resultados.  Discutir e interpretar os dados adequadamente. 
Fase BÔNUS: Deployment e Interface 
• Desenvolver uma página web simples para simular o usuário final inserindo os  dados e recebendo a predição em tempo real. 
3. Critérios de Avaliação 
Para alinhar com o perfil do perfil de ADS, sugiro a seguinte distribuição de pontos: 
• Qualidade do Código (30%): Organização do código, boas práticas de  programação, separação de conceitos (o que é treino fica separado da  interface). 
• Análise Científica/Métricas (40%): Rigor na avaliação dos modelos. O que a  matriz de confusão e demais métricas indicaram foi compreendido e  interpretado corretamente? 
• Relatório e Apresentação (30%): Objetividade, clareza e linguagem adequada  na apresentação do trabalho. 
