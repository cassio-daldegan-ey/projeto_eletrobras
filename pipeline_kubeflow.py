from kfp.v2.dsl import component, pipeline
from kfp.v2 import compiler

@component(
    base_image="us-central1-docker.pkg.dev/elet-dados-gestaodeativos-dev/hvdc-artifacts/hvdc-img:latest"
)
def executar_main_sem_codigo():
    """
    Este componente não faz nada diretamente em Python,
    pois a imagem Docker já tem o ENTRYPOINT que roda o main.py.
    """
    import subprocess
    
    print("Contêiner sendo executado...")
    subprocess.run(["python", "main.py"], check=True)

@pipeline(
    name="pipeline-hvdc",
    description="Exemplo de pipeline chamando um contêiner com ENTRYPOINT configurado."
)
def pipeline_principal():
    executar_main_sem_codigo()

if __name__ == "__main__":
    compiler.Compiler().compile(
        pipeline_func=pipeline_principal,
        package_path="pipeline_kubeflow_v2.json"
    )