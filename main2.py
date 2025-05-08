import requests
import zipfile
import os
from io import BytesIO
# Após executar o download, verifique as imagens
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
#
# # Configurações
# url = "https://cave.cs.columbia.edu/old/databases/SLAM_coil-20_coil-100/coil-20/coil-20-proc.zip"
# save_dir = "./coil20/"
# os.makedirs(save_dir, exist_ok=True)
#
# # Headers para simular navegador
# headers = {
#     "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
# }
#
# try:
#     # Download do arquivo
#     print("Baixando dataset...")
#     response = requests.get(url, headers=headers, stream=True)
#     response.raise_for_status()
#
#     # Salva o arquivo zip
#     zip_path = os.path.join(save_dir, "coil-20-proc.zip")
#     with open(zip_path, "wb") as f:
#         for chunk in response.iter_content(chunk_size=8192):
#             f.write(chunk)
#
#     # Extrai o conteúdo
#     print("Extraindo arquivos...")
#     with zipfile.ZipFile(zip_path, 'r') as zip_ref:
#         zip_ref.extractall(save_dir)
#
#     # Limpeza (opcional)
#     os.remove(zip_path)
#     print("Download e extração concluídos!")
#     print(f"Dataset salvo em: {os.path.abspath(save_dir)}")
#
# except Exception as e:
#     print(f"Erro: {e}")
#     print("Possíveis soluções:")
#     print("1. Verifique sua conexão com a internet")
#     print("2. Tente acessar o link manualmente no navegador")
#     print("3. Verifique as permissões do diretório de destino")


# # Instale a API do Kaggle
# # !pip install kaggle
#
# # Configure suas credenciais (obtenha sua API key em https://www.kaggle.com/settings/account)
# import os
# os.environ['KAGGLE_USERNAME'] = 'luizhcsantos'
# os.environ['KAGGLE_KEY'] = '44843bab094eacfa003487275bd4d625'
#
#
# # Faça o download do dataset
#
#
# # Descompacte e carregue
# import zipfile
# with zipfile.ZipFile('epileptic-seizure-recognition.zip', 'r') as zip_ref:
#     zip_ref.extractall('./seizure_data')
#
# import pandas as pd
# df = pd.read_csv('./seizure_data/Epileptic Seizure Recognition.csv')


import os
import subprocess

# Configurações
KAGGLE_USER = 'luizhcsantos'   # ← Substitua!
KAGGLE_KEY = '44843bab094eacfa003487275bd4d625'  # ← Substitua!
DATASET_ID = 'harunshimanto/epileptic-seizure-recognition'

# Configura ambiente
os.environ['KAGGLE_USERNAME'] = KAGGLE_USER
os.environ['KAGGLE_KEY'] = KAGGLE_KEY

try:
    # Comando de download
    cmd = f'kaggle datasets download -d {DATASET_ID}'
    subprocess.run(cmd, shell=True, check=True)

    print("Download concluído com sucesso!")

except subprocess.CalledProcessError as e:
    print(f"Erro {e.returncode}: {e.output}")
    print("Soluções possíveis:")
    print("1. Verifique username/key em https://www.kaggle.com/settings")
    print("2. Renomeie o dataset se estiver desatualizado")
    print("3. Execute manualmente no terminal: ", cmd)