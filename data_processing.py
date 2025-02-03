"""

Arquivo data_processing, faz o tratatamento dos dados para serem utilizados na previsao.

"""

import re
import json
import itertools
import subprocess
from os import listdir
from os.path import isfile, join
from datetime import datetime, timedelta
import shap as s
import numpy as np
import pandas as pd
from google.cloud import storage
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_percentage_error


def save_dataframe_to_csv(df, file_name):
    try:
        df.to_csv(file_name, index=False, sep=";", decimal=",")
        print(f"DataFrame successfully saved to '{file_name}'")
    except Exception as e:
        print(f"An error occurred while saving the DataFrame: {e}")


def pivot_dataframe_keep_last(df, index_column, column_column, value_column):

    if pd.api.types.is_string_dtype(df[index_column]):
        df[index_column] = pd.to_datetime(df[index_column])

    df = df.drop_duplicates(subset=[index_column, column_column], keep="last")

    pivot_df = df.pivot(index=index_column, columns=column_column, values=value_column)

    pivot_df.columns.name = None
    pivot_df = pivot_df.reset_index()

    return pivot_df


def group_and_fill_by_hour(df, datetime_column):

    df[datetime_column] = pd.to_datetime(df[datetime_column])
    df = df.set_index(datetime_column)
    hourly_max = df.resample("h").max()
    hourly_max = hourly_max.ffill()
    hourly_max = hourly_max.reset_index()
    hourly_max = hourly_max.groupby(pd.Grouper(key=datetime_column,freq='60Min')).first().reset_index()

    return hourly_max


def sage_data_processing() -> pd.DataFrame:
    """A funcao sage_data_processing faz o tratamento dos dados de corrente,
    potencia e tensao enviados pelo SAGE.

    Returns:
        df_sage: Dataframe com dados do SAGE.
    """

    f = open("env.json")
    env = json.load(f)

    files = 1

    path_source = env["PATH_SAGE"]
    
    files = [f for f in listdir(path_source) if isfile(join(path_source, f))]
    
    dfs_sage = []
    for file in files:

        file_path = path_source + file

        with open(file_path, "r", encoding="latin1") as f:
            lines = f.readlines()

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

        df_sage["DATE"] = pd.to_datetime(df_sage["bh_dthr"], format="%Y-%m-%d %H:%M:%S")

        df_sage.drop(columns=["bh_dthr"], inplace=True)
        dfs_sage.append(df_sage)
    
    df_sage = pd.concat(dfs_sage)

    return df_sage


def select_componente(files: list, nome_componente: str) -> list:
    """A funcao select_componente analisa os nomes de todos os arquivos com
    dados de temperatura e selecionar apenas os nomes que contém o nome
    definido na funcao.

    Args:
        files: Lista com o nome de todos os arquivos presentes na pasta inputs.

        nome_componente: String com o componente que estamos procurando nos
        nomes dos arquivos.

    Returns:
        lista_nomes: Lista com os nomes de todos os arquivos nos quais se
        encontra o componente especificado.
    """
    lista_nomes = [
        x
        for x in list(itertools.chain(*[f.split("_") for f in itertools.chain(files)]))
        if nome_componente in x
    ]

    return lista_nomes


def timestamp_to_datetime(variable: pd.DataFrame) -> pd.DataFrame:
    """A funcao timestamp_to_datetime pega os dados em timestamp e transforma
    em datetime.

    Args:
        variable: Variavel com os dados em formato timestamp.

    Returns:
        date_time: Variavel com os dados em formato datetime.
    """

    seconds_since_epoch = variable / 1e7
    dotnet_epoch = datetime(1, 1, 1)
    date_time = dotnet_epoch + timedelta(seconds=seconds_since_epoch)

    return date_time


def replace_seconds_to_zero(string: str) -> str:
    """A funcao replace_seconds_to_zero troca os valores de segundos e milésimos
    de segundos por zero.

    Args:
        string: String de data.

    Returns:
        string: String de data com os valores de segundos substituidos por zero.
    """

    string = string[:-6]
    string = string + ":00:00"

    return string


def base_detalhes(
    df: pd.DataFrame, df_sage: pd.DataFrame, flagCloud: bool
) -> (pd.DataFrame, pd.DataFrame):
    """A funcao base_detalhes gera a base de dados detalhes que iremos
    utilizar no power bi.

    Args:
        df: Base em formato dataframe com o histórico de dados reais de temperatura.

        df_sage: Base de dados com os dados doe SAGE.

    Returns:
        detalhes: Dataframe que contém os dados de temperatura além das variáveis
        potencia, tensao e corrente.

        df_shap: Dataframe que contém os dados a serem utilizados como insumo para
        o modelo de random forest que calcula o SHAP.

        df: Base de dados no formato dataframe com o histórico de dados reais de
        temperatura, filtrado apenas pelas datas para as quais também temos dados
        do SAGE.

    """

    f = open("env.json")
    env = json.load(f)

    df["Data"] = df["Data"].astype(str)
    df["Data"] = df["Data"].apply(replace_seconds_to_zero)

    df["Data"] = pd.to_datetime(df["Data"])

    # Vamos separar o Data entre data e tempo
    df["Time"] = df["Data"].dt.time
    df["Data"] = df["Data"].dt.date

    detalhes = df.copy()
    detalhes["Tipo"] = "Real"

    # Aqui eu tenho que fazer o merge com a base de temperatura, mas eu tenho que
    # tirar aquele tratamento que transforma segundos em zero e que pega apenas
    # a primeira observacao. Tenho que pegar a base como está, sem tratar e fazer
    # o merge das datas.

    df_pivot = pivot_dataframe_keep_last(
        df_sage, index_column="DATE", column_column="descr", value_column="valor"
    )

    df_pivot_grp = group_and_fill_by_hour(df_pivot, datetime_column="DATE")

    # Vamos filtrar apenas as datas para as quais temos dados para as duas
    # bases, de temperatura e do SAGE
    df_pivot_grp.rename(columns={"DATE": "Data"}, inplace=True)

    df_pivot_grp["Data"] = pd.to_datetime(df_pivot_grp["Data"])
    df_pivot_grp["Time"] = df_pivot_grp["Data"].dt.time
    df_pivot_grp["Data"] = df_pivot_grp["Data"].dt.date

    detalhes = pd.merge(detalhes, df_pivot_grp, how="inner", on=["Data", "Time"])
    df_shap = detalhes.copy()

    # Vamos filtrar apenas as variaveis relevantes.

#    detalhes = detalhes.filter(
#        regex="Data|Time|Componente|Tipo|Temperatura|Potencia Ativa|Tensão|Corr|Temperatura"
#    )
    detalhes = detalhes.filter(
        regex="Data|Time|Componente|Tipo|Temperatura|AD-TF1D     Corr Fase B Enr Pri (B)"
    )
    detalhes.columns = detalhes.columns.str.replace('      ', ' ')
    var = list(
        set(list(detalhes.columns))
        ^ set(["Data", "Temperatura", "Componente", "Tipo", "Time"])
    )

    # Vamos gerar a base a ser utilizada para estimar o shap.

    dfs = []
    for x in var:
        df_sage_var = detalhes[
            ["Data", "Temperatura", "Componente", "Tipo", "Time"] + [x]
        ].copy()
        df_sage_var.rename(columns={x: "Max_Valor"}, inplace=True)
        df_sage_var["Variavel"] = x
        dfs.append(df_sage_var)

    detalhes = pd.concat(dfs)

    # Vamos pegar o valor máximo por hora e acrscentar a base.
    df_max = pd.DataFrame(detalhes.groupby(["Data", "Componente"])["Temperatura"].max())
    df_max = df_max.rename(columns={"Temperatura": "Temp Max"})
    detalhes = pd.merge(detalhes, df_max, on=["Data", "Componente"], how="left")

    # Vamos filtrar apenas as datas para as quais nós temos dados
    # em todas as bases.
    for base in [detalhes, df_shap, df]:
        base["Dia-Hora"] = base["Data"].apply(str) + base["Time"].apply(str)
    datas_unicas = list(set(detalhes["Dia-Hora"]))

    for base in [detalhes, df_shap, df]:
        base = base[base["Dia-Hora"].isin(datas_unicas)]
        
    # Por fim, deletamos variaveis que nao serao utilizadas.
    del detalhes["Dia-Hora"]
    del df_shap["Dia-Hora"]
    del df["Dia-Hora"]
    del df_shap["Treino/Teste"]

    if flagCloud:
        # Caminhos local e no GCS
        local_excel_path = "/tmp/base_detalhes.xlsx"
        gcs_excel_path = env["PATH_OUTPUTS"] + "base_detalhes.xlsx"

        # Salva o DataFrame localmente
        detalhes.to_excel(local_excel_path, index=False)

        # Copia o arquivo local para o GCS
        subprocess.run(["gsutil", "cp", local_excel_path, gcs_excel_path], check=True)

    else:
        detalhes.to_excel(env["PATH_OUTPUTS"] + "base_detalhes.xlsx")

    return detalhes, df_shap, df


def train_test(df: pd.DataFrame) -> pd.DataFrame:
    """A funcao train_teste cria uma variavel que epecifica quais observacoes
    em nossa base de dados serao utilizados como base de treino e teste.

    Args:
        df: Base de dados de temperatura.

    Returns:
        df: Base de dados de temperatura acrescida da coluna que especifica
        se cada observacao vai ser de treino ou teste.
    """
    print("Definindo periodos de treino e teste...")

    # Vamos criar uma variavel que define se os dados sao de teste ou de
    # treino. Vamos usar como teste o último dia de dados. O restante
    # é treino.

    dfs = []
    for comp in list(set(df["Componente"])):
        train_test = df[df["Componente"] == comp]
        train_test = train_test.sort_values(by="Data")
        train = train_test.head(len(train_test) - 24*7).copy(deep=True)
        train["Treino/Teste"] = "Treino"
        test = train_test.tail(24*7).copy(deep=True)
        test["Treino/Teste"] = "Teste"
        dfs.append(train)
        dfs.append(test)

    # Vamos unir as bases de treino e teste e ordenar a base.

    df = pd.concat(dfs)
    df = df.sort_values(by=["Componente", "Treino/Teste"], ascending=False)

    return df


def shap_values_func(model, train: pd.DataFrame) -> np.array:
    """A funcao shap_values_func calcula os valores de shap.

    Args:
        model: Modelo de regressao utilizado para gerar os valores de shap.

        train: Dados de treino utilizados no modelo de regressao.

    Returns:
        shap_values: Valores de shap salvos em formato array.
    """
    explainer = s.TreeExplainer(model)
    shap_values = explainer.shap_values(train)
    return shap_values


def feature_selection(df_shap: pd.DataFrame):
    """A funcao feature_selection salva os valores de shap assim como o mape
    e a lista de variáveis utilizadas para todos os modelos possíveis de serem
    estimados dada as variaveis que temos disponiveis.
    
    args:
        df_shap: Dataframe com os dados de temperatura e demais variaveis do sage
        a serem utilizadas nos modelos.
    """
    print("Fazendo o feature selection...")
    f = open("env.json")
    env = json.load(f)

    # Vamos filtrar apenas os Reatores, Tiristores e Bhuchas
    df_shap = df_shap.loc[df_shap["Componente"].str.contains("Reator|Tiristor|Bucha")]

    # Variaveis qua nao serao utilizadas como variaveis x no SHAP
    var = ["Componente", "Tipo", "Data", "Time", "Temperatura"]

    # Vamos separar os dados entre target e features
    y_var = ["Temperatura"]
    x_vars = list(set(list(df_shap.columns)) ^ set(var))
    
    ## Vamos gerar todas as combinacoes possíveis de variaveis para testar os
    ## modelos.
    #vars_combination = []
    #for number in range(len(x_vars)):
    #    list_size = number + 1
    #    vars_list = list(map(list,list(itertools.combinations(x_vars, list_size))))
    #    vars_combination.extend(vars_list)
    
    #print("Combinacao de todas as variaveis...")
    #print(vars_combination)
    
    # Vamos separar os dados entre target e feature, test e train
    vars_combination = [["AD-TF1D     Corr Fase B Enr Pri (B)","AD-TF1      Potencia Ativa (B)"]]
    df = df_shap.copy()
    del df_shap['']
    df = df.dropna(axis = 0, how = 'all')
    
    shap_list = []
    
    for comp in list(set(df["Componente"])):
        for variaveis in vars_combination:
            # Vamos criar as bases de treino e teste
            target_train, target_test = train_test_split(df[df["Componente"] == comp][y_var], test_size=0.2)
            features_train, features_test = train_test_split(df[df["Componente"] == comp][variaveis], test_size=0.2)
            # Vamos calcular os valores de shap tendo como base uma regressao
            # random forest. Mas da para fazer com base em outros algoritmos de
            # regressao
            model = RandomForestRegressor()
            model.fit(features_train,target_train.values.ravel())
            # vamos calcular o mape
            forecast = model.predict(features_test)
            mape = mean_absolute_percentage_error(target_test, forecast)
            # Feita a regressao vamos calcular os valores de shap
            shap_values = shap_values_func(model, features_train)
            shap_values = pd.DataFrame(shap_values,columns=variaveis)
            shap_values = list(shap_values.mean())
            shap_values = pd.DataFrame({'Componente': [comp],'Variaveis': [str(variaveis)],'SHAP_values': [str(shap_values)],'MAPE': [mape]})
            shap_list.append(shap_values)
    
    df_fs = pd.concat(shap_list)
    df_fs.to_excel(env["PATH_OUTPUTS"] + "features_selection.xlsx", index=False)

def export_shap_values(df_shap: pd.DataFrame, flagCloud: bool):
    """A funcao export_shap gera os os dados de shap.

    Args:
        df_shap: Base de dados com o histórico de temperaturas reais e com as
        variaveis utilizadas no feature importance.
    """
    print("Calculando os valores de shap...")

    f = open("env.json")
    env = json.load(f)

    # Vamos filtrar apenas os Reatores, Tiristores e Bhuchas
    df_shap = df_shap.loc[df_shap["Componente"].str.contains("Reator|Tiristor|Bucha")]
    
    # Vamos deletar uma variável errada e modificar alguns nomes
    del df_shap['']

    # Variaveis qua nao serao utilizadas como variaveis x no SHAP
    var = ["Componente", "Tipo", "Data", "Time", "Temperatura"]

    # Vamos separar os dados entre target e features
    y_var = ["Temperatura"]
    x_vars = list(set(list(df_shap.columns)) ^ set(var))
    train = df_shap.copy()
    target_train, features_train = (
        train[y_var + ["Componente"]],
        train[x_vars + ["Componente"]],
    )

    # Vamos calcular os valores de shap para cada um dos componentes
    shap_list = []
    for comp in list(set(features_train["Componente"])):
        base_x = features_train[features_train["Componente"] == comp]
        base_x.reset_index(drop=True, inplace=True)
        base_y = target_train[target_train["Componente"] == comp]
        base_y.reset_index(drop=True, inplace=True)
        # Vamos calcular os valores de shap tendo como base uma regressao
        # random forest. Mas da para fazer com base em outros algoritmos de
        # regressao
        model = RandomForestRegressor()
        model.fit(base_x[x_vars], base_y[y_var].values.ravel())
        # Feita a regressao vamos calcular os valores de shap
        shap_values = shap_values_func(model, base_x[x_vars])
        shap_values = pd.DataFrame(shap_values, columns=[x + "_SHAP" for x in x_vars])
        shap_values.reset_index(drop=True, inplace=True)
        shap_values = pd.concat([shap_values, base_x], axis=1)
        # Calculados os valores de shap vamos acrescentar as demais variaveis
        # que estavam na base detalhes e que vamos utilizar para os filtros
        # do power bi.
        var2 = [x for x in var if x != "Componente"]
        other_variables = df_shap[df_shap["Componente"] == comp][var2].copy()
        other_variables.reset_index(drop=True, inplace=True)
        shap_values = pd.concat([other_variables, shap_values], axis=1)
        # Vamos filtrar apenas o último mes de dados, caso contrário a base
        # fica muito grande
        shap_values = shap_values.tail(24*7)
        # Vamos salvar as bases por componente
        shap_list.append(shap_values)

    # Por fim, concatenamos os shap values para todos os componentes em uma
    # unica base e depois exportar em excel.
    df_shap = pd.concat(shap_list)
    # Vou modificar a formatacao da base para poder utilizar no power bi
    dfs = []
    for x in x_vars:
        new_var = df_shap[var + [x, x + "_SHAP"]].copy()
        new_var["Variável"] = x
        new_var.rename(
            columns={
                x: "valor_real",
                x + "_SHAP": "valor_SHAP",
            },
            inplace=True,
        )
        dfs.append(new_var)

    df_shap = pd.concat(dfs)

    df_shap = df_shap[
        [
            "Componente",
            "Tipo",
            "Data",
            "Time",
            "Temperatura",
            "Variável",
            "valor_real",
            "valor_SHAP",
        ]
    ]
    df_shap.reset_index(drop=True, inplace=True)

    if flagCloud:
        # Caminhos local e no GCS
        local_excel_path = "/tmp/base_shap.xlsx"
        gcs_excel_path = env["PATH_OUTPUTS"] + "base_shap.xlsx"

        # Salva o DataFrame localmente
        df_shap.to_excel(local_excel_path, index=False)

        # Copia o arquivo local para o GCS
        subprocess.run(["gsutil", "cp", local_excel_path, gcs_excel_path], check=True)
    else:
        df_shap.to_excel(env["PATH_OUTPUTS"] + "base_shap.xlsx")

    # Vou exportar também uma base apenas com as datas nao repetidas para
    # fazer um filtro.
    df_datas = df_shap[["Data"]].copy()
    df_datas = df_datas.drop_duplicates()
    df_datas.reset_index(drop=True, inplace=True)

    if flagCloud:
        # Caminhos local e no GCS
        local_excel_path = "/tmp/base_datas.xlsx"
        gcs_excel_path = env["PATH_OUTPUTS"] + "base_datas.xlsx"

        # Salva o DataFrame localmente
        df_datas.to_excel(local_excel_path, index=False)

        # Copia o arquivo local para o GCS
        subprocess.run(["gsutil", "cp", local_excel_path, gcs_excel_path], check=True)
    else:
        df_datas.to_excel(env["PATH_OUTPUTS"] + "base_datas.xlsx")


def rename_valvulas(df: pd.DataFrame) -> pd.DataFrame:
    """A funcao rename_valvulas modifica os nomes das valvulas de retangulo
    ou elipse para os nomes corretos utilizados na sala de valvulas. Tambem
    filtramos apenas tiristores e reatores, pois para os demais equipamentos
    nao fazemos feature importance.

    Args:
        df: Base de dados com os nomes das valvulas que temos nas bases importadas.

    Returns:
        df: Base de dados com os nomes de valvulas corretos.
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
        "Elipse 23"
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
        "Bucha X3"
    ]

    for i in range(len(components_names)):
        df[col] = np.where((df[col] == conditions[i]), components_names[i], df[col])

    # Vamos selecionar apenas o que for reator, tiristor ou para-raio
    df = df[df[col].isin(components_names)]

    return df


def data_importing(data_folder: str, flagCloud: bool) -> pd.DataFrame:
    """A funcao data_import importa as bases de dados e sava em um único dataframe.

    Args:
        data_folder: Caminho para a pasta onde estao os dados enviados pelo hvdc.

    Returns:
        df: DataFrame com os dados de todas as bases enviadas pelo hvdc já tratados
        é consolidados em um único arquivo.
        .
    """
    print("Fazendo a importacao de dados...")

    f = open("env.json")
    env = json.load(f)

    # Criando uma lista com todos os arquivos a serem importados
    if flagCloud:
        # Extrai bucket e prefix do caminho gs://
        match = re.match(r"gs://([^/]+)/(.+)", data_folder)
        if not match:
            raise ValueError(f"Caminho GCS inválido: {data_folder}")
        bucket_name = match.group(1)  # ex: stg-elet-gestaodeativos-vertex-dev
        prefix = match.group(2).rstrip("/") + "/"  # ex: ev_41_hvdc/data/inputs/bases/

        # Lista arquivos no GCS
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=prefix)
        valid_blobs = [blob for blob in blobs if not blob.name.endswith("/")]

        # Gera lista de nomes de arquivo (ex: "arquivo1.csv", etc.)
        files = [blob.name.split("/")[-1] for blob in valid_blobs]
        equip = set(
            select_componente(files, "Elipse")
            + select_componente(files, "Retangulo")
            + select_componente(files, "Ponto")
        )
    else:
        files = [f for f in listdir(data_folder) if isfile(join(data_folder, f))]
        equip = set(
            select_componente(files, "Elipse")
            + select_componente(files, "Retangulo")
            + select_componente(files, "Ponto")
        )

    # Vamos importar os dados do SAGE e pegar as datas para as quais temos dados
    # no SAGE para filtrar os dados de temperatura.

    df_sage = sage_data_processing()

    dfs = []

    # Tratamento de dados base por base
    for i in equip:
        lista = [x for x in files if i in x]
        for file in lista:
            # Importando cada arquivo em dataframe
            df = pd.read_table(data_folder + file, header=None, delimiter=";")
            df["Componente"] = i
            df.reset_index(drop=True, inplace=True)
            df.columns = [
                "Data",
                "Temp Maxima",
                "Temp Media",
                "Temp Minima",
                "Componente",
            ]
            df["Data original"] = df["Data"]
            # Vamos tirar os décimos de segundos
            type(df["Data"].iloc[0])
            # df['Data'] = pd.to_datetime(df['Data']).apply(lambda x: x.replace(microsecond=0))
            ## Transformar os dados de timestamp para datetime
            df["Data"] = df["Data"].apply(timestamp_to_datetime)
            df["Data"] = pd.to_datetime(df["Data"], errors="coerce")
            # Selecionando apenas temperatura maxima
            df = df[["Data", "Temp Maxima", "Componente"]]
            df.rename(columns={"Temp Maxima": "Temperatura"}, inplace=True)
            # Vamos retirar os decimos de segundos da variavel de data
            df["Data"] = df["Data"].apply(
                lambda x: datetime.strptime(str(x).split(".")[0], "%Y-%m-%d %H:%M:%S")
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
    df = train_test(df)
    df = df.reset_index(drop=True)

    # Vamos modificar os nomes das valvulas pelos nomes corretos.
    df = rename_valvulas(df)
    df = df.drop_duplicates()

    # A partir da base com o historico de temperatura, vamos tambem criar a base
    # detalhes, a ser utilizada no power bi.
    # Vamos também filtrar na base temperatura apenas as datas para as quais temos
    # dados para ambas as bases.
    
    detalhes, df_shap, df = base_detalhes(df, df_sage, flagCloud=flagCloud)
    
    # testando feature selection
    
    feature_selection(df_shap)

    # Vamos calcular os shap values

    export_shap_values(df_shap, flagCloud=flagCloud)

    # Por fim, vamos salvar a base de dados.
    if flagCloud:
        # Caminhos local e no GCS
        local_excel_path = "/tmp/base_temperatura.xlsx"
        gcs_excel_path = env["PATH_OUTPUTS"] + "base_temperatura.xlsx"

        # Salva o DataFrame localmente
        df.to_excel(local_excel_path, index=False)

        # Copia o arquivo local para o GCS
        subprocess.run(["gsutil", "cp", local_excel_path, gcs_excel_path], check=True)
    else:
        df.to_excel(env["PATH_OUTPUTS"] + "base_temperatura.xlsx")

    return df
