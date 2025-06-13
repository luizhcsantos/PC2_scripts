import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
import umap.umap_ as umap
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Union, Optional, List
import torch
#from torchvision import datasets, transforms
from PIL import Image
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

class DataProcessor:
    """Encapsula a lógica de pré-processamento para diferentes tipos de dados."""
    def __init__(self):
        self.scaler = StandardScaler()

    def preprocess_numeric(self, data):
        """Trata e escala dados numéricos."""
        print(" - Pré-processando dados numéricos...")
        data_clean = np.nan_to_num(data)
        return self.scaler.fit_transform(data_clean)


class DimensionalityReducer:
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.methods = {
            'pca',
            'tnse',
            'umap'
            'mds'
        }

class ReductionEvaluator:
    """Calcula, armazena e exibe métricas de qualdiade de redução"""
    def __init__(self):
        self.results = []


if __name__ == "__main__":
    processor = DataProcessor()
    reducer = DimensionalityReducer(n_components=2)
    evaluator = ReductionEvaluator()

    methods_to_run = ['pca', 'umap', 'tsne', 'mds']