"""

Arquivo model faz o treino e teste do modelo.

"""

import re
import json
import random
import subprocess
from datetime import date
from ast import literal_eval
import numpy as np
import pandas as pd
from prophet import Prophet
from google.cloud import storage
from sklearn.model_selection import ParameterGrid
from sklearn.metrics import mean_absolute_percentage_error


class ModelTraining:
    """
    A funcao prediction utiliza o algoritmo do Prophet para fazer uma
    previsao da temperatura para cada uma das valvulas.

    Parameters
    ----------
        dfs_ts: list
            Lista com bases de dados de temperatura por componente.
        
        flag_cloud: bool
            Booleano que indica se o modelo irá executar localmente ou no GCP.

    Methods
    -------
        save():
            Método gera listas com os modelos utilizados para previsao de séries
            temporais, bases de treino, bases de teste e nomes dos componentes
            para os quais faremos as previsoes.
    """

    def __init__(self, dfs_ts: list, flag_cloud) -> pd.DataFrame:
        """
        Inicializa a a classe com uma lista dos dataframes utilizados no treno
        dos modelos e um booleano que indica se o modelo irá executar locamente
        ou no GCP.
        """
        self.dfs_ts = dfs_ts
        self.flag_cloud = flag_cloud
    
    @staticmethod
    def _tuning_prophet(
        df: pd.DataFrame, componente: str, flag_cloud: bool
    ) -> (pd.DataFrame, float):
        """
        Nessa funcao fazemos o tunning do modelo prophet e salvamos os
        hiperparametros do modelo que forneceu o melhor mape.

        Parameters
        ----------
            df: pd.DataFrame
                Data frame com os dados de temperatura.

            componente: str
                String que especifica para qual componente estamos fazendo esta previsao.

        Returns
        -------
            test_prev: pd.DataFrame
                Dataframe com os dados de teste e a previsao que geraram o melhor MAPE.

            mape_componente: pd.DataFrame
                Float com o valor de mape referente ao modelo do componente considerado.

        """

        # Importando o arquivo de variaveis de ambiente

        print("Fazendo o tuning do prophet...")

        with open("env.json", encoding="utf-8") as f:
            env = json.load(f)

            if flag_cloud:
                # Baixa o arquivo "tuning.json" do GCS e faz eval
                gcs_path = env["PATH_TUNING"]  # Exemplo: "gs://bucket/pasta/tuning.json"
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

            # Separando os dados entre treino e teste e renomeando as variáveis

            df = df[["ds", "y", "Treino/Teste"]]
            df_train = df[df["Treino/Teste"] == "Treino"][["ds", "y"]]
            df_teste = df[df["Treino/Teste"] == "Teste"][["ds", "y"]]

            # Selecionar os valores utilizados para cada hiperparametro.

            params_grid = {
                "seasonality_mode": ("multiplicative", "additive"),
                "changepoint_prior_scale": [0.01],
                "seasonality_prior_scale": [0.01],
                "holidays_prior_scale": [0.01],
                "n_changepoints": [100],
            }

            # Abaixo, estimamos um modelos para cada uma das diferentes
            # combinacoes de valores para os hiperparametros e salvamos.
            # Como resultado da funcao, vamos retornar apenas os valores
            # de hiperparametros que possibilitaram o menor mape.

            list_test_prev = []

            grid = ParameterGrid(params_grid)
            model_parameters = pd.DataFrame(columns=["MAPE", "Parameters"])
            for p_value in grid:
                print(p_value)
                random.seed(0)
                model = Prophet(
                    changepoint_prior_scale=p_value["changepoint_prior_scale"],
                    holidays_prior_scale=p_value["holidays_prior_scale"],
                    n_changepoints=p_value["n_changepoints"],
                    seasonality_mode=p_value["seasonality_mode"],
                    weekly_seasonality=True,
                    daily_seasonality=True,
                    yearly_seasonality=True,
                    interval_width=0.95,
                )
                model.fit(df_train)
                forecast = model.predict(df_teste)
                mape = mean_absolute_percentage_error(df_teste[["y"]], forecast[["yhat"]])
                list_test_prev.append([df_teste[["ds", "y"]], forecast[["ds", "yhat"]]])
                print("Mean Absolute Percentage Error(MAPE)-------------", mape)
                model_parameters.loc[len(model_parameters)] = pd.Series(
                    {"MAPE": mape, "Parameters": p_value}
                )

            # Vamos salvar a base de teste e a previsao que geraram o maior MAPE
            test, prev = list_test_prev[model_parameters["MAPE"].idxmax()]
            test_prev = pd.merge(test, prev, on=["ds"], how="inner")
            test_prev = test_prev.reset_index()
            test_prev["Componente"] = componente
            test_prev["MAPE"] = (test_prev["y"] - test_prev["yhat"]).abs().div(
                test_prev["y"]
            ).cumsum() / np.arange(1, len(test_prev) + 1)

            # Selecionando os hiperparametros que geraram o menor MAPE
            hiperparametros = model_parameters.sort_values(by=["MAPE"])
            hiperparametros = hiperparametros.reset_index(drop=True)
            hiperparametros.head()

            if tuning[componente]:
                mape_componente = hiperparametros["MAPE"][0]

            # Serializa os hiperparâmetros como JSON
            hiperparametros_json = json.dumps(hiperparametros["Parameters"][0])

            if flag_cloud:
                # Salva localmente no diretório temporário
                local_json_path = f"/tmp/hiperparametros_{componente}.json"
                with open(local_json_path, "w", encoding="utf-8") as f:
                    json.dump(hiperparametros_json, f, ensure_ascii=False, indent=4)

                # Caminho no GCS para salvar o arquivo
                gcs_json_path = env["PATH_OUTPUTS"] + f"hiperparametros_{componente}.json"

                # Usa o gsutil para copiar o arquivo local para o GCS
                subprocess.run(["gsutil", "cp", local_json_path, gcs_json_path], check=True)
            else:
                # Salva diretamente no caminho especificado
                with open(
                    env["PATH_OUTPUTS"] + f"hiperparametros_{componente}.json",
                    "w",
                    encoding="utf-8",
                ) as f:
                    json.dump(hiperparametros_json, f, ensure_ascii=False, indent=4)

            return test_prev, mape_componente

    @staticmethod
    def _hist_mapes_tuning(componente: str, mape_componente: float, flag_cloud):
        """
        A funcao vai fazer uma compilacao com todos os mapes
        dos modelos preditivos. Daqueles para os quais fizemos o tuning na
        ultima execucao e dos quais fizemos tuning anteriormente.

        Args:
            componente: Strig com a informacao de qual componente foi feito
            o tuning do modelo, e para o qual vamos acrescentar um valor de
            mape na base de dados.

            mape_componente: Valor do mape associado ao modelo do componente
            que estamos considerado.
        """

        with open("env.json", encoding="utf-8") as f:
            env = json.load(f)

            mape = {
                "Modelo": [componente],
                "MAPE": [mape_componente],
                "Dia da Estimacao": [date.today()],
            }
            mape = pd.DataFrame(mape)

            # Importando o historico que ja temos sobre os mape
            if flag_cloud:
                local_historico_path = "/tmp/historico_mape.xlsx"
                gcs_historico_path = env["PATH_OUTPUTS"] + "historico_mape.xlsx"

                try:
                    subprocess.run(
                        ["gsutil", "cp", gcs_historico_path, local_historico_path],
                        check=True,
                    )

                    historico = pd.read_excel(local_historico_path)
                except subprocess.CalledProcessError:
                    historico = pd.DataFrame(columns=["Modelo", "MAPE", "Dia da Estimacao"])
            else:
                try:
                    historico = pd.read_excel(env["PATH_OUTPUTS"] + "historico_mape.xlsx")
                except FileNotFoundError:
                    historico = pd.DataFrame(columns=["Modelo", "MAPE", "Dia da Estimacao"])

            historico = historico[["Modelo", "MAPE", "Dia da Estimacao"]]

            # Jutando os dados de historico com os valores de MAPE dos novos tunings
            historico_atualizado = pd.concat([historico, mape])
            historico_atualizado["Dia da Estimacao"] = pd.to_datetime(
                historico_atualizado["Dia da Estimacao"]
            )
            historico_atualizado = historico_atualizado.reset_index(drop=True)
            historico_atualizado = historico_atualizado[
                ["Modelo", "MAPE", "Dia da Estimacao"]
            ]

            # Remove duplicadas (se mesmo dia e mesmo modelo)
            historico_atualizado = historico_atualizado.groupby(
                ["Modelo", "Dia da Estimacao"]
            ).last()
            historico_atualizado = historico_atualizado.sort_values(
                by=["Dia da Estimacao", "Modelo", "MAPE"], ascending=True
            )
            historico_atualizado = historico_atualizado.reset_index()
            historico_atualizado = historico_atualizado[
                ["Modelo", "MAPE", "Dia da Estimacao"]
            ]

            if flag_cloud:
                local_excel_path = "/tmp/historico_mape.xlsx"
                historico_atualizado.to_excel(local_excel_path, index=False)

                subprocess.run(
                    ["gsutil", "cp", local_excel_path, gcs_historico_path], check=True
                )
            else:
                historico_atualizado.to_excel(env["PATH_OUTPUTS"] + "historico_mape.xlsx")

    def save(self):
        """
        Método gera listas com os modelos utilizados para previsao de séries
        temporais, bases de treino, bases de teste e nomes dos componentes
        para os quais faremos as previsoes.
        
        Returns
        -------
        
        models: list
            Lista com os modelos prophet utilizados para previsao.
        dfs_train: list
            Lista com os dataframes de treino dos modelos.
        dfs_test: list
            Lista com os dataframes de teste dos modelos.
        components: set
            Conjunto com a lista de componentes.
        """
        print("Treinando os modelos...")
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

            ls_tp = []
            df_det = pd.DataFrame()

            components = set(pd.concat(self.dfs_ts)["Componente"])

            for df in self.dfs_ts:
                del df["Componente"]

            models = []
            dfs_train = []
            dfs_test = []
            for i in range(len(components)):

                self.dfs_ts[i].rename(
                    columns={"Temperatura": "y", "DateTime": "ds"}, inplace=True
                )
                self.dfs_ts[i]["ds"] = pd.to_datetime(self.dfs_ts[i]["ds"])

                # Vamos fazer o tuning dos modelos por componente

                componente_atual = list(components)[i]
                if tuning[componente_atual]:
                    print(f"Fazendo tuning do modelo para {componente_atual}...")
                    test_prev, mape_componente = ModelTraining._tuning_prophet(
                        self.dfs_ts[i], componente_atual, flag_cloud=self.flag_cloud
                    )
                    ls_tp.append(test_prev)
                    #ModelTraining._hist_mapes_tuning(componente_atual, mape_componente, flag_cloud=flag_cloud)

                # Feito o tuning, vamos pegar os hiperparametros e fazer a previsao
                if self.flag_cloud:
                    hiperparam_path = (
                        env["PATH_OUTPUTS"]
                        + "hiperparametros_"
                        + componente_atual
                        + ".json"
                    )
                    match_hp = re.match(r"gs://([^/]+)/(.+)", hiperparam_path)
                    if not match_hp:
                        raise ValueError(f"Caminho GCS inválido: {hiperparam_path}")
                    bucket_hp = match_hp.group(1)
                    blob_hp = match_hp.group(2)

                    blob2 = storage_client.bucket(bucket_hp).blob(blob_hp)
                    hp_content = blob2.download_as_text()
                    hp = json.loads(hp_content)
                    hp = json.loads(hp)
                else:
                    f = open(
                        env["PATH_OUTPUTS"] + f"hiperparametros_{componente_atual}.json"
                    )
                    hp = json.load(f)
                    hp = json.loads(hp)

                model = Prophet(
                    changepoint_prior_scale=hp["changepoint_prior_scale"],
                    holidays_prior_scale=hp["holidays_prior_scale"],
                    n_changepoints=hp["n_changepoints"],
                    seasonality_mode=hp["seasonality_mode"],
                    seasonality_prior_scale=hp["seasonality_prior_scale"],
                    weekly_seasonality=True,
                    daily_seasonality=True,
                    yearly_seasonality=True,
                    interval_width=0.95,
                )

                print(
                    f"Fazendo treino do modelo e previsao de temperatura para {componente_atual}..."
                )

                # Separa entre TREINO e TESTE (conforme a coluna 'Treino/Teste'):
                df_train = self.dfs_ts[i][
                    self.dfs_ts[i]["Treino/Teste"] == "Treino"
                ].copy()
                df_train["Componente"] = list(components)[i]
                df_test = self.dfs_ts[i][
                    self.dfs_ts[i]["Treino/Teste"] == "Teste"
                ].copy()
                df_test["Componente"] = list(components)[i]

                # Ajuste do modelo somente com o conjunto de TREINO:
                model.fit(df_train[["ds", "y"]])

                # Vamos salvar o modelo
                models.append(model)

                # Vamos salvar as bases de treino e teste
                dfs_train.append(df_train)
                dfs_test.append(df_test)

            # Salvando arquivo detalhado
            detalhado_file = "historico_mape_detalhado.xlsx"
            if self.flag_cloud:
                local_detalhado = "/tmp/" + detalhado_file
                gcs_detalhado = env["PATH_OUTPUTS"] + detalhado_file

                df_det.to_excel(local_detalhado, index=False)

                subprocess.run(
                    ["gsutil", "cp", local_detalhado, gcs_detalhado], check=True
                )
            else:
                local_detalhado = env["PATH_OUTPUTS"] + detalhado_file

                df_det.to_excel(local_detalhado, index=False)

            return models, dfs_train, dfs_test, components
