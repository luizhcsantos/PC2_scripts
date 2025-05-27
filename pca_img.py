import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical  # Updated import for one-hot encoding
from tensorflow.keras.optimizers import RMSprop
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
# Ensure the seaborn library is installed: pip install seaborn
import seaborn as sns
import time
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.metrics import pairwise_distances_chunked
from sklearn.neighbors import NearestNeighbors
import os



def calculate_pairwise_distances_in_chunks(x, metric="euclidean", working_memory=64):
    """
        Função para calcular distâncias de forma eficiente em blocos para evitar problemas de memória.
        """
    print(f"Calculando distancias pairwise em blocos para matriz de tamanho {x.shape}...")

    pairwise_distances_gen = pairwise_distances_chunked(x, metric=metric,working_memory=working_memory, n_jobs=-1)
    chunks = list(pairwise_distances_chunked(x, metric=metric,working_memory=working_memory, n_jobs=-1))

    distant_matrix = np.vstack(chunks)

    # results = []
    # for chunk in pairwise_distances_gen:
    #     results.append(chunk)
    # #distance_matrix = np.vstack(list(pairwise_distances_gen))
    # print("Cálculo de distancia concluído.")
    # return results
    return distant_matrix


def main():
    
    n_execucoes = 5  # Número de execuções do PCA
    k_neighbors = 5  # Número de vizinhos para trustworthiness/continuity
    results = []  # Lista para armazenar resultados

    cache_dir = os.path.join(os.path.expanduser('~'), '.keras', 'datasets')
    if os.path.exists(os.path.join(cache_dir, 'cifar-10-python.tar.gz')):
        os.remove(os.path.join(cache_dir, 'cifar-10-python.tar.gz'))
    
    pic_class = keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = pic_class.load_data()

    # print the shape of training, testing, and label data
    print(type(x_train))
    print('Training Data Shape: ', x_train.shape)
    print('Testing Data Shape: ', x_test.shape)

    print('Label Training Data Shape: ', y_train.shape)
    print('Label Testing Data Shape: ', y_test.shape)

    classes = np.unique(y_train)
    n_classes = len(classes)
    print('Number of Outputs: ', n_classes)
    print('Number of Output Classes: ', classes)

    x_train = x_train/255.0
    print(np.min(x_train), np.max(x_train), x_train.shape)

    sample_ratio = 0.1
    np.random.seed(42)
    n_train_samples = int(len(x_train)*sample_ratio)
    train_indices = np.random.choice(len(x_train), n_train_samples, replace=False)

    x_train = x_train[train_indices]
    y_train = y_train[train_indices]

    x_train_flat = x_train.reshape(-1, 3072) # 32 * 32 * 3
    feat_cols = ['pixel' + str(i) for i in range(x_train_flat.shape[1])]
    df_cifar = pd.DataFrame(x_train_flat, columns=feat_cols)
    df_cifar['Label'] = y_train
    print('Tamanho do DataFrame: {}'.format(df_cifar.shape))
    
    for run in range(n_execucoes):
        # Iniciar medição de tempo
        start_time = time.time()

        pca = PCA(n_components=2)
        principal_components_cifar = pca.fit_transform(
            df_cifar.iloc[:, :-1].values)  # seleciona todas as linhas (:) e todas colunas, menos a última (:-1)
        principal_components_cifar_df = pd.DataFrame(data=principal_components_cifar,
                                                     columns=['Principal Component 1', 'Principal Component 2'])
        principal_components_cifar_df['Label'] = y_train
        # print(principal_components_cifar_df.head())

        # Calcular tempo de execução
        execution_time = time.time() - start_time

        # Calcular Stress usando chunks
        original_data = df_cifar.iloc[:, :-1].values  # Convert to array
        original_distances = calculate_pairwise_distances_in_chunks(original_data, working_memory=1000)

        #Calcular as distancias após o PCA
        pca_distances = calculate_pairwise_distances_in_chunks(principal_components_cifar, working_memory=1000)

        # Calcular o Stress apenas com as médias
        # Defined stress_numerator calculation directly in code
        # stress_numerator = np.sum([(o - p) ** 2 for o, p in zip(original_distances, pca_distances)])
        # stress_denominator = np.sum([o ** 2 for o in original_distances])
        # stress = np.sqrt(stress_numerator / stress_denominator)

        stress_numerator = np.sum((original_distances.flatten() - pca_distances.flatten()) ** 2)
        stress_denominator = np.sum((pca_distances.flatten()) ** 2)
        stress = np.sqrt(stress_numerator / stress_denominator)

        # Calcular Trustworthiness
        trust = trustworthiness(df_cifar.iloc[:, :-1], principal_components_cifar, n_neighbors=k_neighbors)

        # Calcular Continuity
        nbrs_original = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(original_data)
        _, indices_original = nbrs_original.kneighbors(original_data)
        indices_original = indices_original[:, 1:]  # Exclude self

        nbrs_pca = NearestNeighbors(n_neighbors=k_neighbors + 1).fit(principal_components_cifar)
        _, indices_pca = nbrs_pca.kneighbors(principal_components_cifar)
        indices_pca = indices_pca[:, 1:]  # Exclude self

        continuity = 0
        n = df_cifar.shape[0]
        for i in range(n):
            common_neighbors = len(set(indices_original[i]).intersection(set(indices_pca[i])))
            continuity += common_neighbors / k_neighbors
        continuity /= n

        # Salvar resultados
        results.append({
            'conjunto': 'cifar10',
            'execucao': run + 1,
            'tempo_execucao': execution_time,
            'stress': stress,
            'trustworthiness': trust,
            'continuity': continuity
        })

        # Converter os resultados para um DataFrame
        df_results = pd.DataFrame(results)

        # Exibir os resultados no console
        print("\nResumo dos resultados:")
        print(df_results)

        # Salvar os resultados em um arquivo CSV

    caminho_resultados = "resultados/pca/metricas_cifar10.csv"
    df_results = pd.DataFrame(results)

    df_results = pd.DataFrame(results)
    if not os.path.isfile(caminho_resultados):
        # Se o arquivo não existe na pasta, cria com cabeçalho
        df_results.to_csv(caminho_resultados, index=False)
    else:
        # Se o arquivo já existe na pasta, adiciona novas linhas sem repetir o cabeçalho
        df_results.to_csv(caminho_resultados, mode='a', header=False, index=False)

    print(df_results)



if __name__ == '__main__':
    main()
