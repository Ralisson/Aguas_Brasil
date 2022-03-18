
print("Iniciando modelagem para Score Aguas do Brasil, Melhor devedor")
print("    ")

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import os, gc, time, pickle, pyodbc, urllib
import sqlalchemy as sql
from datetime import date, timedelta
from sqlalchemy import create_engine

print("Carregando modelos treinados...")
print("    ")
# Carregando o arquivo do modelo devedores
arquivo_cpf = 'modelo_v1_devedor.sav'
Model = pickle.load(open(arquivo_cpf, 'rb'))
print("Modelo Devedor Carregado!")
   
with open('NameFeature_v1_devedor','rb') as arquivo_cpf:
    nome_features = pickle.load(arquivo_cpf)


#Criando variavei de datas para coleta da query sql

TODAY = date.today()
YESTERDAY = date.today() - timedelta(days=365)
data_inicio = f"{YESTERDAY.year}-{YESTERDAY.month}-{YESTERDAY.day}" #yesterday 
data_final = f"{TODAY.year}-{TODAY.month}-{TODAY.day}"

print("Conectando o bd e coletando dados do mailing...")
print("    ")

params = urllib.parse.quote_plus("DRIVER={SQL Server Native Client 11.0};"
                                 "SERVER=10.10.220.100;"
                                 "DATABASE=dbActyon_GAB;"
                                 "UID=xxxxxxx;"
                                 "PWD=xxxxxxx;"
                                 "Trusted_Connection=no")
engine = sql.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))
engine.connect

query = '''

    SELECT distinct
        tbdevedor.DEVEDOR_ID,
        tbdevedor.CPF,
        tbdevedor.CEP,
        tbdevedor.CONT_ID,
        tbdevedor_fone.FONE,
        tbdevedor_fone.ORIGEM,
        tbdevedor_fone.TIPO,
        tbdevedor.QTDE_TITULOS,
        tbdevedor.VALOR_DIVIDA_ATIVA

    FROM tbdevedor
        JOIN tbdevedor_fone  (NOLOCK) ON tbdevedor.DEVEDOR_ID = tbdevedor_fone.DEVEDOR_ID

    WHERE tbdevedor_fone.STATUS = 0
    AND tbdevedor.DATA_IMPORTACAO between

'''

#mailing = pd.read_sql(query , engine)
mailing = pd.read_sql(query + f"'{data_inicio}' AND '{data_final}'", engine)
print("tamanho da tabela: ", mailing.shape)
engine.dispose()

print("Dados coletados e conexão encerrada")
print("    ")

mailing = mailing.drop_duplicates()
#mailing.dropna(inplace = True)
print("tamanho do mailing: ", mailing.shape)

print("data inicio: ", YESTERDAY)
print(" ")
print("data fim: ", TODAY)
print(" ")

#Criando a coluna de dias de atraso
from datetime import date
TODAY = date.today()
TODAY = pd.to_datetime(TODAY)

mailing.drop_duplicates(inplace=True)
mailing.dropna(inplace=True)
print("Tamanho do mailing pos drop: ", mailing.shape)

#alterando tipo das featrures

mailing['ORIGEM'] = mailing['ORIGEM'].astype('category')
mailing['TIPO'] = mailing['TIPO'].astype('category')
mailing['CEP'] = mailing['CEP'].astype('category')

#Mesma features usadas no machine learning
features = mailing[[ 'QTDE_TITULOS', 'VALOR_DIVIDA_ATIVA', 'CEP','ORIGEM', 'TIPO']] 

#encoder
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
features['CEP']= le.fit_transform(features["CEP"])
features['ORIGEM']= le.fit_transform(features["ORIGEM"])
features['TIPO']= le.fit_transform(features["TIPO"])


from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
features['QTDE_TITULOS'] = scaler.fit_transform(features[['QTDE_TITULOS']])
features['VALOR_DIVIDA_ATIVA'] = scaler.fit_transform(features[['VALOR_DIVIDA_ATIVA']])


print("Aplicando a predição do modelo nos novos dados")
print("    ")

# Aplicação do modelo treinado na nova DEVEDOR
mailing['SCORE_DEVEDOR'] =  Model.predict_proba(features)[:,1]
mailing['PONTUACAO_DEVEDOR'] = ((mailing['SCORE_DEVEDOR'])*10).round()

mailing['DATA_IMPORTACAO'] = TODAY

mailing['BEST_SCORE'] = mailing['SCORE_DEVEDOR']

q1 = mailing.BEST_SCORE.quantile(0.25)
q2 = mailing.BEST_SCORE.quantile(0.50)
q3 = mailing.BEST_SCORE.quantile(0.75)

auxiliar = []
for x in list(mailing['SCORE_DEVEDOR']):
    if 0 <= x < q1:
        auxiliar.append('Improvavel')
    elif q1 <= x < q2:
        auxiliar.append('Pouco Provavel')
    elif q2 <= x < q3:
        auxiliar.append('Provavel')
    elif x >= q3:
        auxiliar.append('Muito Provavel')

mailing['CLASSIFICACAO_DEVEDOR'] = auxiliar 

mailing = mailing[['CPF','DEVEDOR_ID', 'FONE' , 'CONT_ID','SCORE_DEVEDOR', 'CLASSIFICACAO_DEVEDOR','SCORE_CPC', 'CLASSIFICACAO_CPC'
                    ,'BEST_SCORE' ,'CLASSIFICACAO_BEST_SCORE', 'DATA_IMPORTACAO' ]]
mailing = mailing.sort_values(by=['SCORE_CPC'])


print("Media do Score Devedor", mailing.CLASSIFICACAO_DEVEDOR.value_counts())
print(" ")
print("Media do Score CPC", mailing.CLASSIFICACAO_CPC.value_counts())
print(" ")

mailing.drop_duplicates( subset='FONE', keep='last', inplace=True)
print("tamanho do mailing:", mailing.shape)

print("tamanho: ", mailing.shape)
print("Score do mailing Devedor: ", mailing.groupby(['CLASSIFICACAO_DEVEDOR'])['SCORE_DEVEDOR',].mean())
print("Score do mailing CPC: ", mailing.groupby(['CLASSIFICACAO_CPC'])['SCORE_CPC',].mean())
print("    ")

print("tamanho: ", mailing.shape)
print("Classificação do mailing Devedor: ", mailing.groupby(['CLASSIFICACAO_DEVEDOR'])['CPF',].count())
print("Classificação do mailing CPC: ", mailing.groupby(['CLASSIFICACAO_CPC'])['FONE',].count())
print("    ")
print("Exportando Score para o BD MIS...")
print("    ")

mailing_fone = mailing[['FONE' ,'SCORE_CPC', 'CLASSIFICACAO_CPC', 'DATA_IMPORTACAO' ]]
mailing_fone.drop_duplicates(subset=['FONE'], keep='last', inplace=True)
print(mailing_fone.shape)

mailing_cpf = mailing[['CPF','DEVEDOR_ID', 'SCORE_DEVEDOR', 'CLASSIFICACAO_DEVEDOR', 'DATA_IMPORTACAO' ]]
mailing_cpf.drop_duplicates(subset=['CPF'], keep='last', inplace=True)
print(mailing_cpf.shape)

from sqlalchemy.types import Integer, Date, Float, String
dtypes = {
    'CPF': String(),
    'DEVEDOR_ID': String(),
    'CONT_ID': String(),
    'FONE': String(),
    'SCORE_DEVEDOR': Float(),
    'CLASSIFICACAO_DEVEDOR': String(),
    'CLASSIFICACAO_BEST_SCORE': String(),
    'DATA_IMPORTACAO': Date()
}
dtypes_fone = {
    'FONE': String(),
    'DATA_IMPORTACAO': Date()
}
dtypes_cpf = {
    'CPF': String(),
    'DEVEDOR_ID': String(),
    'SCORE_DEVEDOR': Float(),
    'CLASSIFICACAO_DEVEDOR': String(),
    'DATA_IMPORTACAO': Date()
}

se_existir = 'replace'

params = urllib.parse.quote_plus("DRIVER={SQL Server Native Client 11.0};"
                                 "SERVER=10.10.220.100;"
                                 "DATABASE=MIS;"
                                 "UID=xxxxxxx;"
                                 "PWD=xxxxxx;"
                                 "Trusted_Connection=no")
engine = sql.create_engine("mssql+pyodbc:///?odbc_connect={}".format(params))
engine.connect 

mailing.to_sql('DS_GAB_FONE_CPF_Pontuacao', engine, if_exists=se_existir, index=False, dtype = dtypes)
print("Score Aguas do Brasil atualizados em  DS_GAB_FONE_CPF_Pontuacao - BD MIS")

print(" ___________________________________________________________________________________________________________________   ")

mailing_fone.to_sql('DS_GAB_FONE_Pontuacao', engine, if_exists=se_existir, index=False, dtype = dtypes_fone)
print("Score GAB FONE atualizados em  DS_GAB_FONE_Pontuacao - BD MIS")

print(" ___________________________________________________________________________________________________________________   ")

mailing_cpf.to_sql('DS_GAB_CPF_Pontuacao', engine, if_exists=se_existir, index=False, dtype = dtypes_cpf)
print("Score GAB CPF atualizados em  DS_GAB_CPF_Pontuacao - BD MIS")