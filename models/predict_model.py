"""

Arquivo model faz o treino e teste do modelo.

"""

import re
import json
import subprocess
from ast import literal_eval
import numpy as np
import pandas as pd
from google.cloud import storage


class ModelPrediction:
    """
    A classe utiliza o algoritmo do Prophet para fazer uma previsao da temperatura
    para cada uma das valvulas.

    Parameters
    ----------
        models: list
            Lista com os modelos preditivos por componentes.
        dfs_train: list
            Lista com os dataframes de treino.
        dfs_test: list
            Lista com os dataframes de teste
        flag_cloud: bool
            Booleano que indica se o modelo vai executar localmente ou no GCP.

    Returns
    -------
        save():
            Método que gera um dataframe com previsoes de temperatura.
    """

    def __init__(
        self,
        models: list,
        dfs_train: list,
        dfs_test: list,
        components: set,
        flag_cloud: bool,
    ) -> pd.DataFrame:
        """
        A class inicia com listas de modelos preditivos, lista com daraframes de
        treino, dataframes de teste e um booleano que indica se o modelo irá executar
        localmente ou no GCP.
        """
        self.models = models
        self.dfs_train = dfs_train
        self.dfs_test = dfs_test
        self.components = components
        self.flag_cloud = flag_cloud

    def save(self):
        """
        O método gera um dataframe com as previsoes dos modelos de série temporal
        prophet.

        Returns
        -------
        forecasts: pd.DataFrame
            Base de dados com os dados de treino, teste e dados previstos.
        """
        print("Fazendo as previsoes dos modelos...")

        with open("env.json", encoding="utf-8") as f:
            env = json.load(f)

            if self.flag_cloud:
                gcs_path = env["PATH_TUNING"]
                match = re.match(r"gs://([^/]+)/(.+)", gcs_path)
                if not match:
                    raise ValueError(f"Caminho GCS inválido: {gcs_path}")

                bucket_name = match.group(1)
                blob_path = match.group(2)

                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                blob = bucket.blob(blob_path)
                content = blob.download_as_text()
                tuning = json.loads(content)
                tuning = literal_eval(tuning)
            else:
                with open(env["PATH_TUNING"], encoding="utf-8") as f:
                    tuning = json.load(f)
                    tuning = literal_eval(tuning)

            forecasts = []
            df_det = pd.DataFrame()

            for i in range(len(self.components)):
                model = self.models[i]
                componente_atual = list(self.components)[i]
                df_test = self.dfs_test[i]
                df_train = self.dfs_train[i]

                # Vamos gerar as datas para a proxima semana após a ultima data disponivel.
                if not df_test.empty:
                    # pega a última data do teste
                    last_test_date = df_test["ds"].max()
                    start_forecast = last_test_date  # ou last_test_date + 1 dia/hora
                else:
                    # se não existir teste, usa a data do treino
                    start_forecast = df_train["ds"].max()
                print("data de incio da previsao")
                print(start_forecast)

                # A partir daqui, gera datas de previsão (7 dias)
                datas_previsao = []
                for delta in range(24 * 7):
                    datas_previsao.append(start_forecast + np.timedelta64(delta + 1, "h"))

                datas_previsao = pd.DataFrame(datas_previsao, columns=["ds"])

                forecast = model.predict(datas_previsao)
                forecast["Componente"] = componente_atual
                forecast = forecast.rename(columns={"ds": "bh_dthr", "yhat": "Temperatura"})
                forecast = forecast[
                    ["Componente", "bh_dthr", "Temperatura", "yhat_lower", "yhat_upper"]
                ]
                forecasts.append(forecast)

            # Salvando arquivo detalhado
            detalhado_file = "historico_mape_detalhado.xlsx"
            if self.flag_cloud:
                local_detalhado = "/tmp/" + detalhado_file
                gcs_detalhado = env["PATH_OUTPUTS"] + detalhado_file

                df_det.to_excel(local_detalhado, index=False)

                subprocess.run(["gsutil", "cp", local_detalhado, gcs_detalhado], check=True)
            else:
                local_detalhado = env["PATH_OUTPUTS"] + detalhado_file

                df_det.to_excel(local_detalhado, index=False)

            forecasts = pd.concat(forecasts)

            if self.flag_cloud:
                local_excel_path = "/tmp/previsao_temperatura.xlsx"
                forecasts.to_excel(local_excel_path, index=False)
                gcs_excel_path = env["PATH_OUTPUTS"] + "previsao_temperatura.xlsx"
                subprocess.run(
                    ["gsutil", "cp", local_excel_path, gcs_excel_path], check=True
                )
            else:
                forecasts.to_excel(env["PATH_OUTPUTS"] + "previsao_temperatura.xlsx")

            return forecasts
