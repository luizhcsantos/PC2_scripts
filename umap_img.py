# Importando as bibliotecas necessárias
import numpy as np
from keras.datasets import cifar10
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando o conjunto de dados CIFAR-10
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Redimensionando os dados para 2D (mantendo apenas os pixels)
X_train_reshaped = X_train.reshape((X_train.shape[0], -1))

# Normalizando os dados
X_train_normalized = X_train_reshaped / 255.0

print("Forma original dos dados:", X_train.shape)
print("Forma após reshape:", X_train_reshaped.shape)

# Aplicando UMAP (usando apenas uma parte das amostras para exemplo, pois é computacionalmente intensivo)
reducer = umap.UMAP(random_state=42)
embedding = reducer.fit_transform(X_train_normalized[:1000])

print("Forma após UMAP:", embedding.shape)

# Visualizando os resultados
plt.figure(figsize=(10, 8))
scatter = plt.scatter(embedding[:, 0], embedding[:, 1],
                     c=y_train[:5000], cmap='Spectral',
                     s=5, alpha=0.8)
plt.colorbar(scatter)
plt.title('Projeção UMAP do CIFAR-10')
plt.xlabel('UMAP 1')
plt.ylabel('UMAP 2')
plt.show()