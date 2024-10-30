"""

Arquivo main, que executa em sequencias as etapas do modelo, desde o tratamento
de dados até a previsao.

"""

import os
from dotenv import load_dotenv
from data_processing import import_fotos

#Tenho que achar o pacote que salva o comando da pasta onde esta o arquivo
#esse arquivo do main e retornar uma pasta antes.
PATH_PROJETO = "C:/Users/KG858HY/OneDrive - EY/Desktop/projeto_eletrobras/projeto_eletrobras/"

load_dotenv(
    PATH_PROJETO+"env"
)

df_fotos = import_fotos(os.chdir(os.getenv("PATH_FOTOS")))