import json
import pandas as pd
import numpy as np
from pathlib import Path
import subprocess
import os
from datetime import datetime


class Visualization:
    """
    Classe para processar e visualizar os dados.

    A classe permite combinar os DataFrames de dados reais e preditos, gerar a média móvel por grupo
    e salvar os resultados em arquivos CSV.

    Attributes
    ----------
    env : dict
        Parametros carregados a partir do arquivo JSON.
    flag_cloud : bool
        Indica se o processamento ocorrerá em ambiente cloud.

    Methods
    -------
    merging_predictions_actual(df_actual, df_pred, df_sage_input)
        Combina os DataFrames de dados reais e preditos, gera a média móvel por grupo e salva a base consolidada.
    process_data_and_save_alerts(df)
        Filtra os dados previstos, gera os alertas e salva em um arquivo CSV.
    """

    def __init__(self, env_path="env.json", flag_cloud: bool = False):
        """
        Inicializa a classe carregando as configurações do ambiente e definindo a flag de cloud.

        Parameters
        ----------
        env_path : str, optional
            Caminho para o arquivo de configuração (default "env.json").
        flag_cloud : bool, optional
            Indica se o processamento ocorrerá em ambiente cloud.
        """
        with open(env_path, "r") as f:
            self.env = json.load(f)
        self.flag_cloud = flag_cloud

    @staticmethod
    def _aggregate_sage_data_pivot(df_sage_to_agg: pd.DataFrame) -> pd.DataFrame:
        """
        Agrega os dados do DataFrame Sage para obter uma tabela com frequência por dia e hora.

        Parameters
        ----------
        df_sage_to_agg : pd.DataFrame
            DataFrame contendo as colunas 'DATE', 'descr' e 'valor'.

        Returns
        -------
        pd.DataFrame
            DataFrame agregado com as colunas 'Date' e 'Time' para merge, com os valores preenchidos.
        """
        sage_raw = df_sage_to_agg.copy()
        sage_raw["DATE"] = pd.to_datetime(sage_raw["DATE"])

        pivot_sage_data = sage_raw.pivot_table(
            index="DATE", columns="descr", values="valor", aggfunc="first"
        ).reset_index()

        pivot_sage_data["Date"] = pivot_sage_data["DATE"].dt.strftime("%Y-%m-%d")
        pivot_sage_data["Time"] = (
            pivot_sage_data["DATE"].dt.floor("h").dt.strftime("%H:%M:%S")
        )

        pivot_sage_data["Target"] = pivot_sage_data["DATE"].dt.floor("h")
        pivot_sage_data["diff"] = (
            pivot_sage_data["DATE"] - pivot_sage_data["Target"]
        ).abs()

        idx_min = pivot_sage_data.groupby(["Date", "Time"])["diff"].idxmin()
        aggregated_sage = pivot_sage_data.loc[idx_min].copy()

        aggregated_sage.rename(columns={"DATE": "orig_DATE"}, inplace=True)
        aggregated_sage.drop(columns=["Target"], inplace=True)
        aggregated_sage.sort_values("orig_DATE", inplace=True)

        attribute_cols = [
            col
            for col in aggregated_sage.columns
            if col not in ["orig_DATE", "Date", "Time", "diff"]
        ]
        aggregated_sage[attribute_cols] = aggregated_sage[attribute_cols].ffill()

        aggregated_sage.drop(columns=["diff", "orig_DATE"], inplace=True)
        aggregated_sage["Date"] = pd.to_datetime(
            aggregated_sage["Date"], format="%Y-%m-%d"
        ).dt.date
        aggregated_sage["Time"] = pd.to_datetime(
            aggregated_sage["Time"], format="%H:%M:%S"
        ).dt.time

        return aggregated_sage

    @staticmethod
    def _preprocess_dataframe(
        df: pd.DataFrame, df_sage_to_merge: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Realiza o merge do DataFrame principal com o DataFrame de dados SAGE.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame principal com os dados de temperatura.
        df_sage_to_merge : pd.DataFrame
            DataFrame contendo as informações do SAGE.

        Returns
        -------
        pd.DataFrame
            DataFrame após merge.
        """
        aggregated_sage = Visualization._aggregate_sage_data_pivot(df_sage_to_merge)
        df_merged = pd.merge(
            df,
            aggregated_sage,
            left_on=["Data", "Time"],
            right_on=["Date", "Time"],
            how="left",
        )
        df_merged.drop(columns=["Date"], inplace=True)
        df_merged["bh_dthr"] = pd.to_datetime(
            df_merged["Data"].astype(str).str.strip()
            + " "
            + df_merged["Time"].astype(str).str.strip(),
            format="%Y-%m-%d %H:%M:%S",
        )
        return df_merged

    @staticmethod
    def _append_to_csv(df: pd.DataFrame, csv_path: str) -> None:
        """
        Salva o DataFrame em um arquivo CSV, criando ou anexando os dados.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame a ser salvo.
        csv_path : str
            Caminho para o arquivo CSV.
        """
        csv_file = Path(csv_path)
        if not csv_file.exists():
            df.to_csv(csv_file, index=False)
        else:
            df.to_csv(csv_file, mode="a", index=False, header=False)

    @staticmethod
    def _get_7day_moving_average_by_group(
        df,
        coluna_tempo,
        coluna_grupo,
        coluna_valor,
        janela_dias=7,
        nova_coluna="media_7dias_grupo",
        valor_default=np.nan,
    ):
        """
        Calcula a média móvel dos últimos 'janela_dias' para cada grupo.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrada.
        coluna_tempo : str
            Nome da coluna de data/hora.
        coluna_grupo : str
            Nome da coluna de agrupamento.
        coluna_valor : str
            Nome da coluna para calcular a média.
        janela_dias : int
            Número de dias para a janela da média móvel (default é 7).
        nova_coluna : str
            Nome da nova coluna (default é 'media_7dias_grupo').
        valor_default : NaN
            Valor padrão se não houver dados (default é np.nan).

        Returns
        -------
        pd.DataFrame
            DataFrame com a nova coluna adicionada.
        """
        df[coluna_tempo] = pd.to_datetime(df[coluna_tempo])
        data_maxima = df[coluna_tempo].max()
        print("data maxima", data_maxima)
        data_limite = data_maxima - pd.Timedelta(days=janela_dias)
        df_janela = df[df[coluna_tempo] >= data_limite]
        media_grupo = df_janela.groupby(coluna_grupo)[coluna_valor].mean()
        media_grupo_dict = media_grupo.to_dict()
        df[nova_coluna] = df[coluna_grupo].map(media_grupo_dict)
        df[nova_coluna] = df[nova_coluna].fillna(valor_default)
        return df

    @staticmethod
    def _filter_predictions(df):
        """
        Filtra as linhas onde 'Tipo' é 'Previsto'.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame de entrada.

        Returns
        -------
        pd.DataFrame
            DataFrame filtrado.
        """
        print("df de input:", df.head(5))
        return df[df["Tipo"] == "Previsto"].copy()

    @staticmethod
    def _get_first_critical_alerts(df):
        """
        Retorna o primeiro alerta crítico para cada componente.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame contendo as informações de alerta.

        Returns
        -------
        pd.DataFrame
            DataFrame com o primeiro alerta crítico para cada componente.
        """
        df["Alerta_Critico"] = df["Temperatura"] > 65.90
        df_crit = df[df["Alerta_Critico"]].copy()
        df_crit.sort_values("bh_dthr", inplace=True)
        return df_crit.drop_duplicates(subset="Componente", keep="first")

    @staticmethod
    def _get_moving_average_alerts(df):
        """
        Retorna os registros com variação percentual da temperatura acima do limiar.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame contendo os dados de temperatura e média móvel.

        Returns
        -------
        pd.DataFrame
            DataFrame com os registros que acionam o alerta.
        """
        df["Alerta"] = (df["Temperatura"] / df["media_7dias_grupo"] - 1).abs() > 0.168
        return df[df["Alerta"]].copy()

    def process_data_and_save_alerts(self, df: pd.DataFrame) -> None:
        """
        Filtra os dados previstos, gera os alertas e salva-os em um único arquivo CSV.

        Parameters
        ----------
        df : pd.DataFrame
            DataFrame contendo os dados de entrada.
        """
        df_previsto = Visualization._filter_predictions(df)
        df_primeiros_alertas_criticos = Visualization._get_first_critical_alerts(
            df_previsto
        )
        df_alertas_media_movel = Visualization._get_moving_average_alerts(df_previsto)

        df_primeiros_alertas_criticos["Tipo_Alerta"] = "Critico"
        df_alertas_media_movel["Tipo_Alerta"] = "Media_Movel"

        df_alertas_unificado = (
            pd.concat([df_primeiros_alertas_criticos, df_alertas_media_movel])
            .drop_duplicates()
            .reset_index(drop=True)
        )

        df_alertas_unificado.drop(
            columns=["Alerta_Critico", "Alerta"], errors="ignore", inplace=True
        )

        time_of_save = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df_alertas_unificado["Save_DateTime"] = time_of_save

        path_save_alertas = self.env["PATH_OUTPUTS"] + "alertas_hvdc.csv"

        if self.flag_cloud:
            local_csv_path = "/tmp/alertas_hvdc.csv"
            result = subprocess.run(
                ["gsutil", "ls", path_save_alertas],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            if result.returncode == 0:
                subprocess.run(
                    ["gsutil", "cp", path_save_alertas, local_csv_path], check=True
                )
                df_alertas_unificado.to_csv(
                    local_csv_path, mode="a", index=False, header=False
                )
            else:
                df_alertas_unificado.to_csv(
                    local_csv_path, mode="w", index=False, header=True
                )
            subprocess.run(
                ["gsutil", "cp", local_csv_path, path_save_alertas], check=True
            )
        else:
            if os.path.exists(path_save_alertas):
                df_alertas_unificado.to_csv(
                    path_save_alertas, mode="a", index=False, header=False
                )
            else:
                df_alertas_unificado.to_csv(
                    path_save_alertas, mode="w", index=False, header=True
                )

    def merging_predictions_actual(
        self,
        df_actual: pd.DataFrame,
        df_pred: pd.DataFrame,
        df_sage_input: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Combina os DataFrames de dados reais e preditos, gera a média móvel por grupo e salva a base consolidada.

        Parameters
        ----------
        df_actual : pd.DataFrame
            DataFrame com os dados reais.
        df_pred : pd.DataFrame
            DataFrame com os dados preditos.
        df_sage_input : pd.DataFrame
            DataFrame com os dados do Sage.

        Returns
        -------
        pd.DataFrame
            DataFrame combinado com os dados reais e preditos.
        """
        df_actual_with_sage_data = Visualization._preprocess_dataframe(
            df_actual, df_sage_input
        )
        df = pd.concat(
            [
                df_actual_with_sage_data.assign(Tipo="Real"),
                df_pred.assign(Tipo="Previsto"),
            ],
            ignore_index=True,
        )

        condicoes = [
            df["Componente"].str.contains("Tiristor", case=False, na=False),
            df["Componente"].str.contains("Bucha", case=False, na=False),
            df["Componente"].str.contains("Para raio", case=False, na=False),
            df["Componente"].str.contains("Reator", case=False, na=False),
        ]
        grupos = ["Tiristor", "Bucha", "Para-Raio", "Reator"]
        df["Grupo"] = np.select(condicoes, grupos, default="Outro")

        df = Visualization._get_7day_moving_average_by_group(
            df=df,
            coluna_tempo="bh_dthr",
            coluna_grupo="Grupo",
            coluna_valor="Temperatura",
            janela_dias=7,
            nova_coluna="media_7dias_grupo",
            valor_default=np.nan,
        )

        if self.flag_cloud:
            local_csv_path = "/tmp/Dados_hvdc.csv"
            df.to_csv(local_csv_path, index=False)
            gcs_csv_path = self.env["PATH_OUTPUTS"] + "Dados_hvdc.csv"
            subprocess.run(["gsutil", "cp", local_csv_path, gcs_csv_path], check=True)
        else:
            df.to_csv(self.env["PATH_OUTPUTS"] + "Dados_hvdc.csv", index=False)

        return df
