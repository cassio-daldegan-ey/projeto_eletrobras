# Imagem base mínima para Python
FROM python:3.9-slim

# Instala o curl para baixar o Google Cloud SDK
RUN apt-get update && \
    apt-get install -y curl && \
    # Baixa e instala o Google Cloud SDK (necessário para o gsutil)
    curl https://sdk.cloud.google.com | bash -s -- --install-dir=/usr/local --disable-prompts

# Adiciona o Google Cloud SDK ao PATH para que o gsutil funcione
ENV PATH="/usr/local/google-cloud-sdk/bin:${PATH}"

# Copia todo o projeto para o diretório root do container
COPY . /

# Vai para o diretório root
WORKDIR /

# Instala as dependências
RUN pip install --no-cache-dir -r requirements.txt

# Define o comando de entrada
ENTRYPOINT ["python", "main.py"]

