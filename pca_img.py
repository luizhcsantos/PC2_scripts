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
from sklearn.neighbors import NearestNeighbors



def main():
    
    n_execucoes = 1  # Número de execuções do PCA
    k_neighbors = 5  # Número de vizinhos para trustworthiness/continuity
    results = []  # Lista para armazenar resultados
    
    pic_class = keras.datasets.cifar10
    (x_train, y_train), (x_test, y_test) = pic_class.load_data()

    # print the shape of training, testing, and label data
    print('Training Data Shape: ', x_train.shape)
    print('Testing Data Shape: ', x_test.shape)

    print('Label Training Data Shape: ', y_train.shape)
    print('Label Testing Data Shape: ', y_test.shape)

    classes = np.unique(y_train)
    n_classes = len(classes)
    print('Number of Outputs: ', n_classes)
    print('Number of Output Classes: ', classes)


    label_list = {
        0: 'Airplane',
        1: 'Automobile',
        2: 'Bird',
        3: 'Cat',
        4: 'Deer',
        5: 'Dog',
        6: 'Frog',
        7: 'Horse',
        8: 'Ship',
        9: 'Truck'
    }

    x_train = x_train/255.0
    print(np.min(x_train), np.max(x_train), x_train.shape)

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
            df_cifar.iloc[:, :-1])  # seleciona todas as linhas (:) e todas colunas, menos a última (:-1)
        principal_components_cifar_df = pd.DataFrame(data=principal_components_cifar,
                                                     columns=['Principal Component 1', 'Principal Component 2'])
        principal_components_cifar_df['Label'] = y_train
        # print(principal_components_cifar_df.head())

        # Calcular tempo de execução
        execution_time = time.time() - start_time

        # Calcular Stress
        original_distances = pairwise_distances(df_cifar.iloc[:, :-1], metric='euclidean')
        pca_distances = pairwise_distances(principal_components_cifar, metric='euclidean')

        stress_numerator = np.sum((original_distances - pca_distances) ** 2)
        stress_denominator = np.sum(original_distances ** 2)
        stress = np.sqrt(stress_numerator / stress_denominator)

        # Calcular Trustworthiness
        trust = trustworthiness(df_cifar.iloc[:, :-1], principal_components_cifar, n_neighbors=k_neighbors)

        # Calcular Continuity
        nbrs_original = NearestNeighbors(n_neighbors=k_neighbors).fit(df_cifar.iloc[:, :-1])
        _, indices_original = nbrs_original.kneighbors(df_cifar.iloc[:, :-1])

        nbrs_pca = NearestNeighbors(n_neighbors=k_neighbors).fit(principal_components_cifar)
        _, indices_pca = nbrs_pca.kneighbors(principal_components_cifar)

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

        print(results)
    
        # principal_components_cifar_df.to_csv('resultados/pca/pca_cifar10.csv', index=False)
        #
        # print('Variancia explicada: {}'.format(pca.explained_variance_ratio_))
        #
        # plt.figure(figsize=(10, 7))
        # sns.scatterplot(
        #     x="Principal Component 1", y="Principal Component 2",
        #     hue="Label",
        #     palette=sns.color_palette("Set2", 10),
        #     data=principal_components_cifar_df,
        #     legend="full",
        #     alpha=1.0
        # )
        # plt.savefig('pca_cifar10' + run + '.png', dpi=300)
        # #plt.show()



if __name__ == '__main__':
    main()
