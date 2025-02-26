import pandas as pd


class Features:
    """
    A funcao seleciona quais variáveis de nossa base serao utilizadas como features
    para o nosso modelo preditivo.

    Attributes
    ----------
        df: Base de dados com os dados de temperatura para treino e teste.

    Methods
    -------
    time_series(df: pd.DataFrame) -> list
        df: Base de dados com os dados de treino, teste e dados previstos.

    save()
        Salva todos os dataframes utilizados nos modelos preditivos.

    """

    def __init__(self, df):
        """
        Inicializa a classe com a base de dados de temperatura.
        """
        self.df = df

    def time_series(self, df: pd.DataFrame) -> list:
        """
        O método seleciona as variáveis de data e temperatura que serao utilizados
        no nosso modelo de time series, além da variável que permite segmentar os
        períodos entre treino e teste.

        Parameters
        ----------
        df: pd.DataFrame
            Dataframe com os dados de temperatura de todas as válvulas.

        Returns
        -------
        dfs_temp: list
            Lista com as bases de dados. Um por válvula.
        """
        print("Selecionando as variáveis utilizadas no Prophet...")
        dfs_temp = []
        for equip in set(df["Componente"]):
            dfs_temp.append(
                df[df["Componente"] == equip][
                    ["DateTime", "Temperatura", "Treino/Teste", "Componente"]
                ]
            )
        return dfs_temp

    def save(self):
        """
        O método retorna todas as bases de dados que serao utilizadas nos
        modelos preditivos.

        Returns
        -------
        dfs_ts: list
            Lista de dataframes utilizados nos modelos de séries temporais.
        """
        dfs_ts = self.time_series(self.df)
        return dfs_ts
