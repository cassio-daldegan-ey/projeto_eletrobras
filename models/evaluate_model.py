"""

Arquivo model faz o treino e teste do modelo.

"""

import re
import json
import subprocess
from ast import literal_eval
from datetime import datetime
import pandas as pd
from sklearn.metrics import mean_absolute_percentage_error
from google.cloud import storage


class ModelEvaluation:
    """
    A funcao utiliza o algoritmo do Prophet para fazer uma previsao da temperatura
    para cada uma das valvulas.

    Attributes
    ----------
        models: list
            Lista dos modelos preditivos utilizados por componente.
        dfs_test: list
            Lista com os dataframes de treino por componente.
        dfs_train: list
            Lista com os dataframes de teste por componente.
        flag_cloud: list
            Booleano que indica se o modelo vai executar localmente ou no GCP.

    Methods
    -------
        save():
            Método que salva as previsões dos modelos preditivos.

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
        Inicia a classe com listas de modelos, dataframes de treino, dataframes
        de teste e um booleano que indica se o modelo vai executar localmente
        ou no GCP.
        """
        self.models = models
        self.dfs_train = dfs_train
        self.dfs_test = dfs_test
        self.flag_cloud = flag_cloud
        self.components = components

    def save(self):
        """
        O método salva as bases de dados de avaliação dos modelos, com os valores
        de MAPE.
        """
        print("Calculando o MAPE dos modelos...")
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

            df_det = pd.DataFrame()

            for df in self.dfs_train:
                del df["Componente"]

            for i in range(len(self.components)):
                df_test = self.dfs_test[i]
                model = self.models[i]
                componente_atual = list(self.components)[i]

                # Separa entre TREINO e TESTE (conforme a coluna 'Treino/Teste'):

                # Caso exista conjunto de TESTE, calculamos MAPE e salvamos num arquivo
                if not df_test.empty:
                    forecast_test = model.predict(df_test[["ds"]])

                    # Calcula MAPE
                    mape_pred = (
                        mean_absolute_percentage_error(df_test["y"], forecast_test["yhat"])
                        * 100
                    )
                    print(f"MAPE de previsão para {componente_atual} = {mape_pred:.2f}%")

                    # Monta o registro
                    new_entry = pd.DataFrame(
                        {
                            "Modelo": [componente_atual],
                            "MAPE": [mape_pred],
                            "Dia da Estimacao": [datetime.now()],
                        }
                    )

                    # Tenta ler o histórico existente (local ou GCS)
                    historico_pred_file = "historico_mape_pred.xlsx"
                    if self.flag_cloud:
                        local_pred_path = "/tmp/" + historico_pred_file
                        gcs_pred_path = env["PATH_OUTPUTS"] + historico_pred_file
                        try:
                            subprocess.run(
                                ["gsutil", "cp", gcs_pred_path, local_pred_path], check=True
                            )
                            df_pred_hist = pd.read_excel(local_pred_path)
                        except subprocess.CalledProcessError:
                            df_pred_hist = pd.DataFrame(
                                columns=["Modelo", "MAPE", "Dia da Estimacao"]
                            )
                    else:
                        local_pred_path = env["PATH_OUTPUTS"] + historico_pred_file
                        try:
                            df_pred_hist = pd.read_excel(local_pred_path)
                        except FileNotFoundError:
                            df_pred_hist = pd.DataFrame(
                                columns=["Modelo", "MAPE", "Dia da Estimacao"]
                            )

                    # Concatena, remove duplicados e ordena
                    df_pred_hist = pd.concat([df_pred_hist, new_entry], ignore_index=True)
                    df_pred_hist["Dia da Estimacao"] = pd.to_datetime(
                        df_pred_hist["Dia da Estimacao"]
                    )
                    df_pred_hist = (
                        df_pred_hist.groupby(["Modelo", "Dia da Estimacao"], as_index=False)
                        .last()
                        .sort_values(by=["Dia da Estimacao", "Modelo"], ascending=True)
                    )

                    # Salva
                    df_pred_hist.to_excel(local_pred_path, index=False)
                    if self.flag_cloud:
                        subprocess.run(
                            ["gsutil", "cp", local_pred_path, gcs_pred_path], check=True
                        )

                    # Salvando um SEGUNDO arquivo de MAPE DETALHADO, com todas as linhas sem agrupar

                    # Cria o DataFrame df_merged
                    df_merged = pd.merge(
                        df_test[["ds", "y"]],
                        forecast_test[["ds", "yhat"]],
                        on="ds",
                        how="inner",
                    )

                    df_merged["MAPE"] = (
                        (df_merged["y"] - df_merged["yhat"]).abs() / df_merged["y"] * 100
                    )
                    df_merged["Componente"] = componente_atual

                    # Ajusta colunas finais
                    df_merged.rename(
                        columns={
                            "ds": "Data",
                            "y": "Temperatura_Real",
                            "yhat": "Temperatura_Previsto",
                        },
                        inplace=True,
                    )
                    df_merged = df_merged[
                        [
                            "Componente",
                            "Data",
                            "Temperatura_Previsto",
                            "Temperatura_Real",
                            "MAPE",
                        ]
                    ]

                    # Se quiser ordenar:
                    df_merged["Data"] = pd.to_datetime(df_merged["Data"])
                    df_merged.sort_values(
                        by=["Componente", "Data"], ascending=True, inplace=True
                    )

                    df_det = pd.concat([df_det, df_merged], ignore_index=True)
                else:
                    print(
                        f"Não há dados de teste para {componente_atual}. MAPE não calculado."
                    )

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