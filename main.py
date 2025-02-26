import time
import os
import sys
from pipeline import Pipeline

start_time = time.time()

src_path = os.path.abspath(os.path.join(os.getcwd(), "../..", "hvdc"))
sys.path.append(src_path)

# Cria uma instância do Pipeline e executa as etapas do modelo
pipeline = Pipeline(env_path="env.json")
df_output_hvdc = pipeline.run()

end_time = time.time()
elapsed_time = end_time - start_time
minutes = int(elapsed_time // 60)
seconds = elapsed_time % 60
print(f"Tempo de execução: {minutes} min {seconds:.2f} s")
