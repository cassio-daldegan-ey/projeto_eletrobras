"""
Esse script faz o processamento dos dados.
"""

import json
import subprocess
from datetime import datetime
import shap as s
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


class Processing:
    """
    Uma classe que faz o processamento dos dados.

    Attributes
    ----------
    df: pd.DataFrame
        Base de dados de temperatura.

    df_sage: pd.DataFrame
        Base de dados do SAGE.

    flag_cloud: pd.DataFrame
        Booleano que indica se o modelo vai executar localmente ou no GCP.

    Methods
    -------
    base_detalhes(df,df_sage,flag_cloud) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame)
        Gera a base de dados detalhes, que será utilizada para as estatísticas descritivas.

    export_shap_values(df_shap,flag_cloud)
        Gera a base de dados com os valores de shap values para serem utilizadas
        em estatística descritiva.

    save()
        Salva as bases de temperatura e SAGE.

    """

    def __init__(self, df: pd.DataFrame, df_sage: pd.DataFrame, flag_cloud: bool):
        """
        Inicializa a classe com um dataframe de temperatura, um dataframe do SAGE
        e um boleano que indica se o modelo irá executar localmente ou no GCP.
        """
        self.df = df
        self.df_sage = df_sage
        self.flag_cloud = flag_cloud

    @staticmethod
    def _rename_valvulas(df: pd.DataFrame) -> pd.DataFrame:
        """
        A funcao modifica os nomes das valvulas de retangulo ou elipse para os nomes
        corretos utilizados na sala de valvulas. Tambem filtramos apenas tiristores
        e reatores, pois para os demais equipamentos nao fazemos feature importance.

        Parameters
        ----------
            df: pd.DataFrame
                Base de dados com os nomes das valvulas que temos nas bases importadas.

        Returns
        -------
            df: pd.DataFrame
                Base de dados com os nomes de valvulas corretos.
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
    def _replace_seconds_to_zero(string: str) -> str:
        """
        A funcao troca os valores de segundos e milésimos de segundos por zero.

        Parameters
        ----------
            string: str
                String de data.

        Returns
        -------
            string: str
                String de data com os valores de segundos substituidos por zero.
        """

        string = string[:-6]
        string = string + ":00:00"

        return string

    @staticmethod
    def _pivot_dataframe_keep_last(
        df: pd.DataFrame, index_column: str, column_column: str, value_column: str
    ) -> pd.DataFrame:
        """
        A funcao faz o pivot da base de dados.

        Parameters
        ----------
            df: pd.DataFrame
                Base de dados que irá sofrer o pivot.

            index_column: str
                String com as colunas utilizadas como index.

            column_columns: str
                Nome da coluna.

            value_column: str
                Nome da coluna.

        Returns
        -------
            pivot_df: pd.DataFrame
                Base de dados após o pivot.
        """

        if pd.api.types.is_string_dtype(df[index_column]):
            df[index_column] = pd.to_datetime(df[index_column])

        df = df.drop_duplicates(subset=[index_column, column_column], keep="last")

        pivot_df = df.pivot(
            index=index_column, columns=column_column, values=value_column
        )

        pivot_df.columns.name = None
        pivot_df = pivot_df.reset_index()

        return pivot_df

    @staticmethod
    def _group_and_fill_by_hour(df: pd.DataFrame, datetime_column: str) -> pd.DataFrame:
        """
        Funcao que trata os missings da base de dados.

        Parameters
        ----------
            df: pd.DataFrame
                Base de dados do SAGE para a qual vamos tratar os missings.

            datatime_column: str
                Nome da variável datetime.

        Returns
        -------
            hourly_max: pd.DataFrame
                Base de dados após o tratamento dos missings.
        """
        df[datetime_column] = pd.to_datetime(df[datetime_column])
        df = df.set_index(datetime_column)
        hourly_max = df.resample("h").max()
        hourly_max = hourly_max.ffill()
        hourly_max = hourly_max.reset_index()

        return hourly_max

    @staticmethod
    def _shap_values_func(model, train: pd.DataFrame) -> np.array:
        """
        A funcao calcula os valores de shap.

        Parameters
        ----------
            model:
                Modelo de regressao utilizado para gerar os valores de shap.

            train: pd.DataFrame
                Dados de treino utilizados no modelo de regressao.

        Returns
        -------
            shap_values: np.array
                Valores de shap salvos em formato array.
        """
        explainer = s.TreeExplainer(model)
        shap_values = explainer.shap_values(train)
        return shap_values

    @staticmethod
    def _sage_expected_values(df_pivot_grp: pd.DataFrame) -> pd.DataFrame:
        """
        A funcao calcula os valores de shap.

        Parameters
        ----------
            df_pivot_grp: pd.DataFrame
                DataFrame com valores do SAGE já com tratamento.

        Returns
        -------
            df_sage_expected: pd.DataFrame
                Dataframe com os valores experados do SAGE.
        """
        df_sage_expected = df_pivot_grp.copy()

        df_sage_expected.loc[:, "Hora"] = df_sage_expected["Data"].dt.hour
        df_sage_expected.loc[:, "Minuto"] = df_sage_expected["Data"].dt.minute
        df_sage_expected.loc[:, "Segundo"] = df_sage_expected["Data"].dt.second
        df_sage_expected.loc[:, "Mes"] = df_sage_expected["Data"].dt.month
        df_sage_expected.loc[:, "Dia"] = df_sage_expected["Data"].dt.day

        df_sage_expected = pd.DataFrame(
            df_sage_expected.groupby(["Hora", "Minuto", "Segundo", "Mes", "Dia"])[
                df_sage_expected.columns.difference(
                    ["Hora", "Minuto", "Segundo", "Mes", "Dia"]
                )
            ].median()
        )

        df_sage_expected = df_sage_expected.rename_axis(
            ["Hora", "Minuto", "Segundo", "Mes", "Dia"]
        ).reset_index()

        df_sage_expected_ano2 = df_sage_expected.copy()
        df_sage_expected.loc[:, "Ano"] = datetime.now().year
        df_sage_expected_ano2["Data"] = df_sage_expected_ano2[
            "Data"
        ] + pd.offsets.DateOffset(years=1)
        df_sage_expected = pd.concat([df_sage_expected, df_sage_expected_ano2])
        df_sage_expected = df_sage_expected.drop(
            ["Ano", "Mes", "Dia", "Hora", "Minuto", "Segundo"], axis=1
        )

        return df_sage_expected

    def base_detalhes(
        self, df, df_sage, flag_cloud
    ) -> (pd.DataFrame, pd.DataFrame, pd.DataFrame):
        """A funcao base_detalhes gera a base de dados detalhes que iremos
        utilizar no power bi.

        Parameters
        ----------
            df: pd.DataFrame
                Base em formato dataframe com o histórico de dados reais de temperatura.

            df_sage: pd.DataFrame
                Base de dados com os dados doe SAGE.

            flag_cloud: bool
                Boleano que indica se o modelo vai rodar localmente ou no GCP.

        Returns
        -------
            df_shap: pd.DataFrame
                Dataframe que contém os dados a serem utilizados como insumo para
                o modelo de random forest que calcula o SHAP.

            df: pd.DataFrame
                Base de dados no formato dataframe com o histórico de dados reais
                de temperatura, filtrado apenas pelas datas para as quais também
                temos dados do SAGE.

        """
        print("Gerando a base detalhes...")
        with open("env.json", encoding="utf-8") as f:
            env = json.load(f)

            df["Data"] = df["Data"].astype(str)
            df["Data"] = df["Data"].apply(Processing._replace_seconds_to_zero)

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

            df_pivot = Processing._pivot_dataframe_keep_last(
                df_sage,
                index_column="DATE",
                column_column="descr",
                value_column="valor",
            )

            df_pivot_grp = Processing._group_and_fill_by_hour(
                df_pivot, datetime_column="DATE"
            )

            # Vamos filtrar apenas as datas para as quais temos dados para as duas
            # bases, de temperatura e do SAGE
            df_pivot_grp.rename(columns={"DATE": "Data"}, inplace=True)

            df_pivot_grp["Data"] = pd.to_datetime(df_pivot_grp["Data"])

            # Acrescentar aqui a base de dados de valores esperados do SAGE
            df_sage_expected = Processing._sage_expected_values(df_pivot_grp)

            df_pivot_grp["Time"] = df_pivot_grp["Data"].dt.time
            df_pivot_grp["Data"] = df_pivot_grp["Data"].dt.date

            detalhes = pd.merge(
                detalhes, df_pivot_grp, how="inner", on=["Data", "Time"]
            )
            df_shap = detalhes.copy()

            # Vamos filtrar apenas as variaveis relevantes.

            del detalhes["Treino/Teste"]
            detalhes.columns = detalhes.columns.str.replace("      ", " ")
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
            df_max = pd.DataFrame(
                detalhes.groupby(["Data", "Componente"])["Temperatura"].max()
            )
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

            df["DateTime"] = pd.to_datetime(
                df["Data"].astype(str) + " " + df["Time"].astype(str),
                format="%Y-%m-%d %H:%M:%S",
            )

            if flag_cloud:
                # Caminhos local e no GCS
                local_csv_path = "/tmp/base_detalhes.xlsx"
                gcs_csv_path = env["PATH_OUTPUTS"] + "base_detalhes.csv"

                # Salva o DataFrame localmente
                detalhes.to_csv(local_csv_path, index=False)

                # Copia o arquivo local para o GCS
                subprocess.run(
                    ["gsutil", "cp", local_csv_path, gcs_csv_path], check=True
                )

            else:
                detalhes.to_csv(env["PATH_OUTPUTS"] + "base_detalhes.csv", index=False)

        return df_shap, df, df_sage_expected

    def export_shap_values(self, df_shap, flag_cloud):
        """
        A funcao export_shap gera os os dados de shap.

        Parameters
        ----------
            df_shap: pd.DataFrame
                Base de dados com o histórico de temperaturas reais e com as
                variaveis utilizadas no feature importance.
        """
        print("Calculando os valores de shap...")

        with open("env.json", encoding="utf-8") as f:
            env = json.load(f)

            # Vamos filtrar apenas os Reatores, Tiristores e Bhuchas
            df_shap = df_shap.loc[
                df_shap["Componente"].str.contains("Reator|Tiristor|Bucha")
            ]

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
                shap_values = Processing._shap_values_func(model, base_x[x_vars])
                shap_values = pd.DataFrame(
                    shap_values, columns=[x + "_SHAP" for x in x_vars]
                )
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
                shap_values = shap_values.tail(
                    24 * 7
                )  # COMENTAR ESTA LINHA QUANDO FILTRAR AS COLUNAS DO SAGE
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

            if flag_cloud:
                # Caminhos local e no GCS
                local_excel_path = "/tmp/base_shap.xlsx"
                gcs_excel_path = env["PATH_OUTPUTS"] + "base_shap.xlsx"

                # Salva o DataFrame localmente
                df_shap.to_excel(local_excel_path, index=False)

                # Copia o arquivo local para o GCS
                subprocess.run(
                    ["gsutil", "cp", local_excel_path, gcs_excel_path], check=True
                )
            else:
                df_shap.to_excel(env["PATH_OUTPUTS"] + "base_shap.xlsx")

            # Vou exportar também uma base apenas com as datas nao repetidas para
            # fazer um filtro.
            df_datas = df_shap[["Data"]].copy()
            df_datas = df_datas.drop_duplicates()
            df_datas.reset_index(drop=True, inplace=True)

            if flag_cloud:
                # Caminhos local e no GCS
                local_excel_path = "/tmp/base_datas.xlsx"
                gcs_excel_path = env["PATH_OUTPUTS"] + "base_datas.xlsx"

                # Salva o DataFrame localmente
                df_datas.to_excel(local_excel_path, index=False)

                # Copia o arquivo local para o GCS
                subprocess.run(
                    ["gsutil", "cp", local_excel_path, gcs_excel_path], check=True
                )
            else:
                df_datas.to_excel(env["PATH_OUTPUTS"] + "base_datas.xlsx")

    @staticmethod
    def _alerts_limits(
        df_shap: pd.DataFrame, df_sage_expected: pd.DataFrame, flag_cloud: bool
    ):
        """A funcao alerts_limits gera os limites de temperatura a partir do qual será
        gerado um alerta.

        Parameters
        ----------
            df_shap: pd.DataFrame
                DataFrame com valores do SAGE.

            df_sage_expected: pd.DataFrame
                Dataframe com os valores eperados do sage para o futuro.

            flag_cloud: bool
                Booleano que determina se o modelo executa localmente ou no GCP.
        """
        print("Gerando os limites para os alertas...")

        with open("env.json", encoding="utf-8") as f:
            env = json.load(f)

            # Vamos filtrar apenas os Reatores e Tiristores
            df_shap = df_shap.loc[
                df_shap["Componente"].str.contains("Reator|Tiristor")
            ].copy()
            df_shap.loc[:, "Reator|Tiristor"] = np.where(
                df_shap["Componente"].str.contains("Reator"), "Reator", "Tiristor"
            )
            del df_shap["Componente"]

            # Vamos prever apenas datas posteriores ao fim da base real
            max_data = df_shap["Data"].iloc[-1]

            # Variaveis qua nao serao utilizadas como variaveis x no SHAP
            var = ["Reator|Tiristor", "Tipo", "Data", "Time", "Temperatura"]

            # Vamos acrescentar variaveis de controle para mes e periodo chuvoso
            for base in [df_shap, df_sage_expected]:
                base["Data"] = pd.to_datetime(base["Data"], errors="coerce").copy()
                base.loc[:, "Mes"] = base["Data"].dt.month
                base.loc[:, "Periodo Chuvoso"] = str(1)
                base.loc[:, "Periodo Chuvoso"] = (base["Periodo Chuvoso"]).case_when(
                    [((base["Mes"] >= 3) & (base["Mes"] <= 11), "0")]
                )
                base["Mes"] = base["Mes"].astype(str)
                meses = pd.get_dummies(base["Mes"])
                base = base.drop("Mes", axis=1)
                base = pd.concat([base, meses], axis=1)

            # Vamos listar as variaveis que entram em target e features
            y_var = ["Temperatura"]
            x_vars = list(set(list(df_shap.columns)) ^ set(var))

            # Vamos calcular a previsao de temperatura para cada tipo de componente
            temp_pred = []
            for comp in list(set(df_shap["Reator|Tiristor"])):
                train = (
                    df_shap[df_shap["Reator|Tiristor"] == comp]
                    .head(len(df_shap) - 1)
                    .copy(deep=True)
                )
                features_train = train[x_vars]
                target_train = train[y_var]
                model = RandomForestRegressor()
                model.fit(features_train, target_train.values.ravel())
                prediction = model.predict(
                    df_sage_expected[df_sage_expected["Data"].dt.date > max_data][
                        x_vars
                    ]
                )
                df_comp = df_sage_expected[
                    df_sage_expected["Data"].dt.date > max_data
                ].copy()
                df_comp.loc[:, "Comp"] = comp
                df_pred = pd.DataFrame(
                    {
                        "Data": df_sage_expected[
                            df_sage_expected["Data"].dt.date > max_data
                        ]["Data"],
                        "Tipo": df_comp["Comp"],
                        "Temperatura esperada": prediction,
                    }
                )
                temp_pred.append(df_pred)

            # Por fim, criamos o dataframe com os valores de limite de temperatura
            limite_alertas = pd.concat(temp_pred)

            if flag_cloud:
                # Caminhos local e no GCS
                local_excel_path = "/tmp/limite_alertas.xlsx"
                gcs_excel_path = env["PATH_OUTPUTS"] + "limite_alertas.xlsx"

                # Salva o DataFrame localmente
                limite_alertas.to_excel(local_excel_path, index=False)

                # Copia o arquivo local para o GCS
                subprocess.run(
                    ["gsutil", "cp", local_excel_path, gcs_excel_path], check=True
                )
            else:
                limite_alertas.to_excel(env["PATH_OUTPUTS"] + "limite_alertas.xlsx")

    def save(self):
        """
        A funcao salva as bases com dados de temperatura e do SAGE.

        Returns
        -------

            df: pd.DataFrame
                Base de dados de temperatura.

            df_shap: pd.DataFrame
                Base de dados com o histórico de temperaturas reais e com as
                variaveis utilizadas no feature importance.
        """
        with open("env.json", encoding="utf-8") as f:
            env = json.load(f)

            # A partir da base com o historico de temperatura, vamos tambem criar a base
            # detalhes, a ser utilizada no power bi.
            # Vamos também filtrar na base temperatura apenas as datas para as quais temos
            # dados para ambas as bases.

            df_shap, df, df_sage_expected = self.base_detalhes(
                self.df, self.df_sage, flag_cloud=self.flag_cloud
            )

            # Gerar a série de valores esperados
            Processing._alerts_limits(
                df_shap, df_sage_expected, flag_cloud=self.flag_cloud
            )

            # Vamos calcular os shap values

            # self.export_shap_values(df_shap, flag_cloud=self.flag_cloud)

            if self.flag_cloud:
                local_excel_path = "/tmp/base_temperatura.xlsx"
                gcs_excel_path = env["PATH_OUTPUTS"] + "base_temperatura.xlsx"

                df.to_excel(local_excel_path, index=False)

                subprocess.run(
                    ["gsutil", "cp", local_excel_path, gcs_excel_path], check=True
                )
            else:
                df.to_excel(env["PATH_OUTPUTS"] + "base_temperatura.xlsx")

        return df, self.df_sage
