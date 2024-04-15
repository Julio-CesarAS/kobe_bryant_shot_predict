import pandas as pd
import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt


# Em uma aplicação real, o ideal é que isso aqui seria um artefato salvo, sendo baixado direto do mlflow
prod_file = '../data/processed/prediction_prod.parquet'
dev_file = '../data/processed/prediction_test.parquet'

# Titulo da Sidebar
st.sidebar.title('Painel de Controle')
st.sidebar.markdown(f"""
Apresentação do resultado.
""")

data_prod = pd.read_parquet(prod_file)
data_dev = pd.read_parquet(dev_file)

# st.write(data_prod)
# st.write(data_dev)


fignum = plt.figure(figsize=(8, 6))
# Saida do modelo dados dev
sns.distplot(data_dev.prediction_score_1,
            label='Teste',
            ax=plt.gca())

# Saida do modelo dados prod
sns.distplot(data_prod.predict_score,
            label='Producão',
            ax=plt.gca())

# Pergunta: Tem muita diferença entre produção e teste
plt.title('Monitoramento do Desvio de Dados da Saída do Modelo')
plt.ylabel('Densidade Estimada')
plt.xlabel('Probabilidade de Alta Qualidade')
plt.xlim((0, 1))
plt.grid(True)
plt.legend(loc='best')


st.pyplot(fignum)



# Apresentando o resultado - Item 9
from sklearn import metrics 

st.write(metrics.classification_report(data_dev.shot_made_flag, data_dev.prediction_label))

