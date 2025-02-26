"""
Esse script carrega os dados de temperatura e do SAGE utilizados no modelo.
"""

import re
import json
import itertools
from os import listdir
from os.path import isfile, join
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
from google.cloud import storage


class DataImporting:
    """
    Uma classe que promove a importação de dados.

    Attributes
    ----------
    data_folder: pd.DataFrame
        Caminho para a pasta onde estao os dados enviados pelo hvdc.

    Methods
    -------
    temperatura(data_folder:str,equip:list,files:list,flag_cloud:bool) -> pd.DataFrame
        Gera a base de dados com os valores de temperatura.

    sage(flag_cloud:bool) -> pd.DataFrame
        Gera a base de dados com os valores do SAGE.

    save_export() -> (pd.DataFrame,pd.DataFrame)
        Exporta e salva as bases de dados de temperatura e do SAGE.
    """

    def __init__(self, data_folder: str, flag_cloud: bool):
        """
        Inicializa a classe com o caminho para a pasta onde estão salvos os valores
        de temperatura e o booleano que indica se o modelo vai ser executado localmente
        ou no GCP.
        """
        self.data_folder = data_folder
        self.flag_cloud = flag_cloud
    
    @staticmethod
    def _select_componente(files: list, nome_componente: str) -> list:
        """
        A funcao analisa os nomes de todos os arquivos com
        dados de temperatura e selecionar apenas os nomes que contém o nome
        definido na funcao.

        Parameters
        ----------
            files: list
                Lista com o nome de todos os arquivos presentes na pasta inputs.

            nome_componente: str
                String com o componente que estamos procurando nos nomes dos arquivos.

        Returns
        -------
            lista_nomes: list
                Lista com os nomes de todos os arquivos nos quais se encontra o componente
                especificado.
        """
        lista_nomes = [
            x
            for x in list(itertools.chain(*[f.split("_") for f in itertools.chain(files)]))
            if nome_componente in x
        ]

        return lista_nomes

    @staticmethod
    def _select_max_column(df: pd.DataFrame) -> pd.DataFrame:
        """
        A funcao seleciona a temperatura que apresenta os valores
        de temperatura maxima entre as variaveis de nome 1, 2 e 3. Ficam apenas as
        as variaveis componente e de valor maximo.

        Parameters
        ----------
            df: pd.DataFrame
                Base de dados com todos os valores de temperatura.

        Returns
        -------
            df: pd.DataFrame
                Base de dados com os valores de temperatura maxima.
        """
        for name in [1, 2, 3]:
            df[name] = df[name].astype(float)

        if (max(df[1]) > max(df[2])) & (max(df[1]) > max(df[3])):
            df = df[[0, 1, "Componente"]]
        elif (max(df[2]) > max(df[1])) & (max(df[2]) > max(df[3])):
            df = df[[0, 2, "Componente"]]
        elif (max(df[3]) > max(df[1])) & (max(df[3]) > max(df[2])):
            df = df[[0, 3, "Componente"]]

        df.columns = [
            "Data",
            "Temp Maxima",
            "Componente",
        ]
        return df

    @staticmethod
    def _timestamp_to_datetime(variable: pd.DataFrame) -> pd.DataFrame:
        """
        A funcao pega os dados em timestamp e transforma
        em datetime.

        Parameters
        ----------
        variable: pd.DataFrame
            Variavel com os dados em formato timestamp.

        Returns
        -------
        date_time: pd.DataFrame
            Variavel com os dados em formato datetime.
        """

        seconds_since_epoch = variable / 1e7
        dotnet_epoch = datetime(1, 1, 1)
        date_time = dotnet_epoch + timedelta(seconds=seconds_since_epoch)

        return date_time

    @staticmethod
    def _rename_valvulas(df: pd.DataFrame) -> pd.DataFrame:
        """
        A funcao modifica os nomes das valvulas de retangulo ou elipse para os nomes
        corretos utilizados na sala de valvulas. Tambem filtramos apenas tiristores
        e reatores, pois para os demais equipamentos nao fazemos feature importance.

        Parameters
        ----------
        df: pd.DataFrame
            Base de dados de temperatura.

        Returns
        -------
        df: pd.DataFrame
            Base de dados de temperatura com os nomes corrigidos.
        """

        col = "Componente"
        conditions = [
            "Elipse 2",
            "Retangulo 3",
            "Retangulo 4",
            "Elipse 7",
            "Elipse 5",
            "Elipse 6",
            "Retangulo 1",
            "Retangulo 0",
            "Ponto 8",
            "Ponto 9",
            "Elipse 20",
            "Elipse 21",
            "Elipse 22",
            "Elipse 23",
        ]
        components_names = [
            "Modulo Tiristor V1.A3",
            "Reator V2.A3.L3",
            "Reator V2.A7.L7",
            "Modulo Tiristor V1.A8",
            "Modulo Tiristor V1.A7",
            "Modulo Tiristor V1.A4",
            "Reator V1.A3.L3",
            "Reator V1.A7.L7",
            "Bucha X1",
            "Bucha X2",
            "Para raio X1.Y1",
            "Para raio X2.Y1",
            "Para raio X3.Y1",
            "Bucha X3",
        ]

        for i, component_name in enumerate(components_names):
            df[col] = np.where((df[col] == conditions[i]), component_name, df[col])

        # Vamos selecionar apenas o que for reator, tiristor ou para-raio
        df = df[df[col].isin(components_names)]

        return df

    @staticmethod
    def _train_test(df: pd.DataFrame) -> pd.DataFrame:
        """
        A funcao cria uma variavel que epecifica quais observacoes
        em nossa base de dados serao utilizados como base de treino e teste.

        Parameters
        ----------
            df: pd.DataFrame
                Base de dados de temperatura.

        Returns
        -------
            df: pd.DataFrame
                Base de dados de temperatura acrescida da coluna que especifica se
                cada observacao vai ser de treino ou teste.
        """
        print("Definindo periodos de treino e teste...")

        # Vamos criar uma variavel que define se os dados sao de teste ou de
        # treino. Vamos usar como teste o último dia de dados. O restante
        # é treino.

        dfs = []
        for comp in list(set(df["Componente"])):
            traintest = df[df["Componente"] == comp]
            traintest = traintest.sort_values(by="Data")
            train = traintest.head(len(traintest) - 24 * 7).copy(deep=True)
            train["Treino/Teste"] = "Treino"
            test = traintest.tail(24 * 7).copy(deep=True)
            test["Treino/Teste"] = "Teste"
            dfs.append(train)
            dfs.append(test)

        # Vamos unir as bases de treino e teste e ordenar a base.

        df = pd.concat(dfs)
        df = df.sort_values(by=["Componente", "Treino/Teste"], ascending=False)

        return df

    def temperatura(
        self, data_folder: str, equip: list, files: list
    ) -> pd.DataFrame:
        """
        Gera a base de dados da temperatura da sala de válvulas.

        Parameters
        ----------
        data_folder: str
            Caminho para a base de dados onde os dados estao salvos.

        equip: list
            Lista com todos os componentes monitorados pelas câmeras térmicas.

        files: list
            Lista com os arquivos importados para gerar a base de dados de temperatura.

        Returns
        -------
        df: pd.DataFrame
            Base de dados com os valores de temperatura.
        """
        print("Importando dados das câmeras térmicas...")
        dfs = []

        # Tratamento de dados base por base
        for i in equip:
            lista = [x for x in files if i in x]
            for file in lista:
                # Importando cada arquivo em dataframe
                df = pd.read_table(data_folder + file, header=None, delimiter=";")
                df["Componente"] = i
                df.reset_index(drop=True, inplace=True)
                # Vamos selecionar dentre as colunas de temperatura apenas a maxima
                df = DataImporting._select_max_column(df)
                df["Data original"] = df["Data"]
                # Vamos tirar os décimos de segundos
                type(df["Data"].iloc[0])
                # df['Data'] = pd.to_datetime(df['Data']).apply(lambda x: x.replace(microsecond=0))
                ## Transformar os dados de timestamp para datetime
                df["Data"] = df["Data"].apply(DataImporting._timestamp_to_datetime)
                df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
                # Selecionando apenas temperatura maxima
                df = df[["Data", "Temp Maxima", "Componente"]]
                df.rename(columns={"Temp Maxima": "Temperatura"}, inplace=True)
                # Vamos retirar os decimos de segundos da variavel de data
                df["Data"] = df["Data"].apply(
                    lambda x: datetime.strptime(
                        str(x).split(".", maxsplit=1)[0], "%Y-%m-%d %H:%M:%S"
                    )
                )
                # Vamos extrair os dados de hora
                df["Hora"] = df["Data"].apply(lambda x: x.to_pydatetime())
                df["Hora"] = df["Hora"].dt.hour
                df["Dia"] = df["Data"].apply(lambda x: x.to_pydatetime())
                df["Dia"] = df["Dia"].dt.date
                # Por fim, vamos filtrar apenas a primeiro dado de temperatura disponível
                # para cada hora.
                df = df.sort_values(["Componente", "Data", "Hora", "Temperatura"])
                df = df.groupby(["Dia", "Hora"]).first().reset_index()
                df = df[["Data", "Componente", "Temperatura"]]
                dfs.append(df)

        # Vamos juntar todas as bases em uma única e separar entre dados de
        # treino e teste
        df = pd.concat(dfs)
        df = DataImporting._train_test(df)
        df = df.reset_index(drop=True)

        # Vamos modificar os nomes das valvulas pelos nomes corretos.
        df = DataImporting._rename_valvulas(df)
        df = df.drop_duplicates()

        return df

    def sage(self, flag_cloud) -> pd.DataFrame:
        """
        Gera a base de dados do SAGE.

        Parameters
        ----------
        flag_cloud: bool
            Booleano que indica se o script vai executar a versão local ou do GCP.

        Returns
        -------
        df: pd.DataFrame
            Base de dados com os valores do SAGE.
        """
        print("Importando dados do SAGE...")
        with open("env.json", encoding="utf-8") as f:
            env = json.load(f)

            path_source = env["PATH_SAGE"]

            if flag_cloud:
                # Modo GCS. ex.: path_source = "gs://meu-bucket/pasta_sage/"
                match = re.match(r"gs://([^/]+)/(.+)", path_source)
                if not match:
                    raise ValueError(f"Caminho GCS inválido: {path_source}")

                bucket_name = match.group(1)
                prefix = match.group(2).rstrip("/") + "/"

                storage_client = storage.Client()
                bucket = storage_client.bucket(bucket_name)
                # Lista todos os blobs
                blobs = bucket.list_blobs(prefix=prefix)
                files = [blob for blob in blobs if not blob.name.endswith("/")]
            else:
                files = [f for f in listdir(path_source) if isfile(join(path_source, f))]

            dfs_sage = []
            for file in files:

                if flag_cloud:
                    all_text = file.download_as_text(encoding="latin1")
                    lines = all_text.splitlines(True)
                else:
                    file_path = path_source + file
                    with open(file_path, "r", encoding="latin1") as f_local:
                        lines = f_local.readlines()

                data_lines = lines[3:]

                data = []
                for line in data_lines:
                    if line.strip():

                        row = [
                            line[:35].replace("|", "").strip(),
                            line[35:77].replace("|", "").strip(),
                            line[78:97].replace("|", "").strip(),
                            line[106:].replace("|", "").strip(),
                        ]
                        data.append(row)

                columns = ["id", "descr", "bh_dthr", "valor"]

                df_sage = pd.DataFrame(data, columns=columns)

                df_sage["valor"] = pd.to_numeric(df_sage["valor"], errors="coerce")

                df_sage = df_sage[["descr", "bh_dthr", "valor"]]

                colunas_analise = [
                    "AD-TF1      Potencia Ativa (A)",
                    "AD-TF1D     Potencia Ativa (A)",
                    "AD-TF1Y     Potencia Ativa (A)",
                    "AD-TF1      Potencia Reativa (A)",
                    "AD-TF1D     Potencia Reativa (A)",
                    "AD-TF1Y     Potencia Reativa (A)",
                    "AD-TF1D     Corr Fase B Enr Pri (A)",
                    "AD-TF1Y     Corr Fase B Enr Pri (A)",
                    "AD-TF1D     Temperatura Óleo do Topo FA",
                    "AD-TF1D     Temperatura Óleo do Topo FB",
                    "AD-TF1D     Temperatura Óleo do Topo FV",
                    "AD-TF1D     Temperatura Óleo da Base FA",
                    "AD-TF1D     Temperatura Óleo da Base FB",
                    "AD-TF1D     Temperatura Óleo da Base FV",
                    "AD-TF1Y     Temperatura Óleo do Topo FA",
                    "AD-TF1Y     Temperatura Óleo do Topo FB",
                    "AD-TF1Y     Temperatura Óleo do Topo FV",
                    "AD-TF1Y     Temperatura Óleo da Base FA",
                    "AD-TF1Y     Temperatura Óleo da Base FB",
                    "AD-TF1Y     Temperatura Óleo da Base FV",
                ]

                df_sage = df_sage[df_sage["descr"].isin(colunas_analise)]

                df_sage["DATE"] = pd.to_datetime(
                    df_sage["bh_dthr"], format="%Y-%m-%d %H:%M:%S"
                )

                df_sage.drop(columns=["bh_dthr"], inplace=True)
                dfs_sage.append(df_sage)

        df_sage = pd.concat(dfs_sage)
        return df_sage

    def save(self) -> (pd.DataFrame, pd.DataFrame):
        """
        Salva as bases de dados de temperatura e do SAGE.

        Returns
        -------
        df: pd.DataFrame
            Base de dados com os valores de temperatura.

        df_sage: pd.DataFrame
            Base de dados com os valores do SAGE.
        """
        # Criando uma lista com todos os arquivos a serem importados
        if self.flag_cloud:
            # Extrai bucket e prefix do caminho gs://
            match = re.match(r"gs://([^/]+)/(.+)", self.data_folder)
            if not match:
                raise ValueError(f"Caminho GCS inválido: {self.data_folder}")
            bucket_name = match.group(1)  # ex: stg-elet-gestaodeativos-vertex-dev
            prefix = (
                match.group(2).rstrip("/") + "/"
            )  # ex: ev_41_hvdc/data/inputs/bases/

            # Lista arquivos no GCS
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blobs = bucket.list_blobs(prefix=prefix)
            valid_blobs = [blob for blob in blobs if not blob.name.endswith("/")]

            # Gera lista de nomes de arquivo (ex: "arquivo1.csv", etc.)
            files = [blob.name.split("/")[-1] for blob in valid_blobs]
            equip = set(
                DataImporting._select_componente(files, "Elipse")
                + DataImporting._select_componente(files, "Retangulo")
                + DataImporting._select_componente(files, "Ponto")
            )
        else:
            files = [
                f
                for f in listdir(self.data_folder)
                if isfile(join(self.data_folder, f))
            ]
            equip = set(
                DataImporting._select_componente(files, "Elipse")
                + DataImporting._select_componente(files, "Retangulo")
                + DataImporting._select_componente(files, "Ponto")
            )

        # Vamos importar os dados do SAGE e pegar as datas para as quais temos dados
        # no SAGE para filtrar os dados de temperatura.

        df_sage = self.sage(self.flag_cloud)

        df = self.temperatura(self.data_folder, equip, files)

        return df, df_sage
