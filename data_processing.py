"""

Funcoes utilizadas para o tratamento dos dados.

"""

import flirpy
import pandas as pd

def import_fotos(PATH_FOTOS: str) -> pd.DataFrame:
    """A funcao import_fotos faz a importacao das fotos com imagens térmicas
    e transforma em dataframe para utilizarmos como inputs dos modelos
    preditivos.
    
    Args:
        
        PATH_FOTOS: Caminho para a pasta onde estao salvas as fotos termograficas
        
    Outputs:
        
        df_temp: Base de dados com os dados de temperatura de todas as imagens.
        
    """
    
    df = 1
    
    return df