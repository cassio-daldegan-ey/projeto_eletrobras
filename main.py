"""

Arquivo main, que executa em sequencias as etapas do modelo, desde o tratamento
de dados até a previsao.

"""

import os
from dotenv import load_dotenv

load_dotenv(
    "C:/Users/KG858HY/EY/Projeto Eletrobras/src_eletrobras/projeto_eletrobras/env"
)

os.chdir(os.getenv("PATH_SRC"))

from data_processing import (
    extracao_dados,
    )

df = extracao_dados(os.getenv("PATH_VIDEOS"))

