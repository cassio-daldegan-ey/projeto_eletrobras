# pipeline.py
import json
from data.load_data import DataImporting
from data.process_data import Processing
from features.build_features import Features
from models.evaluate_model import ModelEvaluation
from models.predict_model import ModelPrediction
from models.train_model import ModelTraining
from visualization.visualize import Visualization

class Pipeline:
    def __init__(self, env_path="env.json"):
        # Lê as configurações do ambiente
        with open(env_path, "r") as f:
            self.env = json.load(f)
        self.flag_cloud = self.env["FLAG_CLOUD"]
        
        self.df = None
        self.df_sage_data = None
        self.prev = None
        self.df_output_hvdc = None

    def run(self):
        # Importa e trata os dados de temperatura
        self.df, self.df_sage_data = DataImporting(self.env["PATH_INPUTS"], flag_cloud=self.flag_cloud).save()

        # Processamento dos dados
        self.df, self.df_sage_data = Processing(self.df, self.df_sage_data, self.flag_cloud).save()
        
        # Seleciona as variaveis independentes do modelo
        #dfs_ts = Features(self.df).save()
        
        # Treinar o modelo
        #models, dfs_train, dfs_test, components = ModelTraining(dfs_ts, self.flag_cloud).save()
        
        # Avaliar o modelo
        #ModelEvaluation(models, dfs_train, dfs_test, components, self.flag_cloud).save()
        
        # Previsao
        #self.prev = ModelPrediction(models, dfs_train, dfs_test, components, self.flag_cloud).save()
        
        # Gera  arquivo consolidado e arquivo de alertas
        #vis = Visualization(flag_cloud=self.flag_cloud)

        #self.df_output_hvdc = vis.merging_predictions_actual(self.df, self.prev, self.df_sage_data)

        #vis.process_data_and_save_alerts(self.df_output_hvdc)
        
        return self.df_output_hvdc
