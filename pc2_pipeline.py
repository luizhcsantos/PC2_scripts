import os
import time

import numpy as np
import pandas as pd
import csv
from kagglehub import datasets
from numpy.ma.core import indices, append
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE, MDS
import umap
import matplotlib.pyplot as plt
import seaborn as sns
import re
from typing import Union, Optional, List, Tuple, Any
# import torch
# from torchvision import datasets
from sklearn.metrics import pairwise_distances
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sympy.solvers.diophantine.diophantine import sum_of_squares

# Tenta importar o UMAP e define uma flag baseada no sucesso
# try:
#     import umap.umap_ as umap
#     UMAP_AVAILABLE = True
# except AttributeError as e:
#     print(f"AVISO: Não foi possível importar a biblioteca UMAP. A análise com UMAP será pulada. Erro: {e}")
#     UMAP_AVAILABLE = False


np.random.seed(42)
sns.set_theme(style='whitegrid', context='notebook', rc={'figure.figsize': (10, 8)})


class DataProcessor:
    """Encapsula a lógica de pré-processamento para diferentes tipos de dados."""

    def __init__(self):
        self.scaler = StandardScaler()
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = nltk.corpus.stopwords.words("english")


    def _clean_text_document(self, doc: str) -> str:
        doc = doc.lower()
        doc = re.sub(r"[^a-z\s]", '', doc)
        tokens = word_tokenize(doc)
        cleaned_tokens = [
            self.lemmatizer.lemmatize(word) for word in tokens if word not in self.stopwords and len(word) > 2
        ]
        return " ".join(cleaned_tokens)

    def preprocess_numeric(self, data: pd.DataFrame) -> np.ndarray:
        """Trata e escala dados numéricos."""
        print(" - Pré-processando dados numéricos...")
        data_numeric = data.select_dtypes(include=np.number)
        data_categorical = data.select_dtypes(exclude=np.number)
        if not data_numeric.empty:
            data_categorical_encoded = pd.get_dummies(data_categorical, drop_first=True)
            processed_data = pd.concat([data_numeric, data_categorical_encoded], axis=1)
        else:
            processed_data = data_numeric

        data_clean = np.nan_to_num(processed_data.astype(float))
        return self.scaler.fit_transform(data_clean)

    def preprocess_image(self, images: np.ndarray) -> np.ndarray:
        print("  - Pré-processando dados de imagem...")
        images_normalized = images.astype(float) / 255.0
        images_reshaped = images_normalized.reshape(images.shape[0], -1)
        return self.scaler.fit_transform(images_reshaped)

    def preprocess_text(self, texts: List[str]) -> np.ndarray:
        """Limpa e vetoriza uma lista de documentos de texto."""
        print("  - Pré-processando dados de texto...")

        print("    - Limpando textos (removendo stopwords, lematizando, etc.)...")
        cleaned_texts = [self._clean_text_document(doc) for doc in texts]

        # print("    - Amostra do texto após a limpeza (primeiras 5 mensagens):")
        # for i, text in enumerate(cleaned_texts[:5]):
        #     print(f"      {i + 1}: '{text}'")

        print("    - Vetorizando textos com TF-IDF...")

        vectorizer = TfidfVectorizer(max_features=200, min_df=1, max_df=0.8)

        tfidf_matrix = vectorizer.fit_transform(cleaned_texts).toarray()
        return tfidf_matrix
        # Se o vocabulário ainda estiver vazio, a linha acima falhará.
        # if tfidf_matrix.shape[1] == 0:
        #     print("ALERTA: O vocabulário ainda está vazio mesmo com min_df=1. Retornando matriz vazia.")
        #     return np.array([[] for _ in range(tfidf_matrix.shape[0])])
        #
        # return vectorizer.fit_transform(tfidf_matrix)

class DimensionalityReducer:
    def __init__(self, n_components: int = 2):
        self.n_components = n_components
        self.methods = {
            'pca': PCA(n_components=self.n_components),
            'tsne': TSNE(n_components=self.n_components, random_state=42, max_iter=300),
            'umap': umap.UMAP(n_components=self.n_components, random_state=42),
            'mds': MDS(n_components=self.n_components, random_state=42, n_jobs=-1, normalized_stress='auto')
        }
        # if UMAP_AVAILABLE:
        #     self.methods['umap'] = umap.UMAP(n_components=self.n_components, random_state=42)

    def reduce(self, data: np.ndarray, method: str) -> np.ndarray:
        if method.lower() not in self.methods:
            raise ValueError(f"Método {method} não suprotado")

        reducer = self.methods[method.lower()]
        print(data)

        if method.lower() == 'tsne':
            reducer.set_params(perplexity=max(5, min(30, data.shape[0] - 1)))
        elif method.lower() == 'umap':
            reducer.set_params(n_neighbors=min(5, min(15, data.shape[0] - 1)))

        return reducer.fit_transform(data)

class ReductionEvaluator:
    """Calcula, armazena e exibe métricas de qualdiade de redução"""

    def __init__(self):
        self.results = []

    def _calculate_stress(self, x_high: np.ndarray, x_low: np.ndarray) -> float:

        with np.errstate(divide='ignore', invalid='ignore'):
            d_high = pairwise_distances(x_high)
            d_low = pairwise_distances(x_low)
            sum_of_squares_high = np.sum(d_high ** 2)
            if sum_of_squares_high == 0: return 0.0
            stress = np.sqrt(np.sum((d_high - d_low) ** 2) / sum_of_squares_high)
        return stress

    def _calculate_trustworthiness_continuity(self, x_high: np.ndarray, x_low: np.ndarray, n_neighbors: int) -> Tuple[
        float, float]:
        n_samples = x_high.shape[0]
        k = min(n_neighbors, n_samples - 1)
        if k == 0: return 1.0, 1.0

        dist_high = pairwise_distances(x_high)
        dist_low = pairwise_distances(x_low)
        ind_high = np.argsort(dist_high, axis=1)[:, 1:k + 1]
        ind_low = np.argsort(dist_low, axis=1)[:, 1:k + 1]

        high_neighbors = {i: set(ind_high[i]) for i in range(n_samples)}
        low_neighbors = {i: set(ind_low[i]) for i in range(n_samples)}

        trust = 0.0
        for i in range(n_samples):
            u_i = low_neighbors[i] - high_neighbors[i]
            for j in u_i:
                rank = np.where(np.sort(dist_high[i]) == dist_high[i, j])[0][0]
                trust += (rank - k)

        cont = 0.0
        for i in range(n_samples):
            v_i = high_neighbors[i] - low_neighbors[i]
            for j in v_i:
                rank = np.where(np.sort(dist_low[i]) == dist_low[i, j])[0][0]
                cont += (rank - k)

        norm = 2 / (n_samples * k * (2 * n_samples - 3 * k - 1))
        return (1 - norm * trust), (1 - norm * cont)

    def evaluate(self, x_high: np.ndarray, x_low: np.ndarray, dataset_name: str, method_name: str, exec_time: float):
        print(f"  - Avaliando {method_name.upper()}...")
        k = min(7, x_high.shape[0] - 1)
        trust, cont = self._calculate_trustworthiness_continuity(x_high, x_low, n_neighbors=k)
        stress = self._calculate_stress(x_high, x_low)

        metrics = {
            'Dataset': dataset_name, 'Method': method_name.upper(), 'Time (s)': exec_time,
            'Stress': stress, 'Trustwrthiness': trust, 'Continuity': cont
        }

        self.results.append(metrics)

    def get_results_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self.results)

    @staticmethod
    def plot_results(x_reduced: np.ndarray, dataset_name: str, method_name: str, plots_dir: str,
                     labels: Optional[np.ndarray] = None):
        """
        Plota o resultado e salva a imagem.
        """
        plt.figure(figsize=(10, 8))
        title = f"Projeção para '{dataset_name}' usando {method_name.upper()}"

        #  Forçada para Numérico ---
        if labels is not None:
            # Converte os rótulos para um tipo numérico. Se algum não puder ser
            # convertido, será transformado em 'NaN' (Not a Number).
            numeric_labels = pd.to_numeric(labels, errors='coerce')

            # Filtra quaisquer pontos cujos rótulos não puderam ser convertidos
            valid_indices = ~np.isnan(numeric_labels)
            numeric_labels = numeric_labels[valid_indices].astype(int)
            x_reduced_valid = x_reduced[valid_indices]

            scatter = plt.scatter(x_reduced_valid[:, 0], x_reduced_valid[:, 1], c=numeric_labels, cmap='viridis', s=15,
                                  alpha=0.7)

            if len(np.unique(numeric_labels)) <= 15:
                plt.legend(handles=scatter.legend_elements(num=len(np.unique(numeric_labels)))[0],
                           labels=[str(l) for l in np.unique(numeric_labels)], title="Classes")
        else:
            # Se não houver rótulos, plota tudo de uma cor
            scatter = plt.scatter(x_reduced[:, 0], x_reduced[:, 1], s=15, alpha=0.7)
        # -----------------------------------------------------------------

        plt.title(title, fontsize=16)
        plt.xlabel(f"{method_name.upper()} Componente 1")
        plt.ylabel(f"{method_name.upper()} Componente 2")
        plt.grid(True)

        dataset_slug = re.sub(r'[^a-z0-9_]', '', dataset_name.lower().replace(' ', '_'))
        filename = f"{dataset_slug}_{method_name.lower()}.png"
        full_path = os.path.join(plots_dir, filename)
        plt.savefig(full_path, dpi=300, bbox_inches='tight')
        print(f"    - Gráfico salvo em: {full_path}")

        plt.show()


def load_cifar10(path: str, subsample: int = 1000) -> Tuple[Any, Any, str]:
    train_set = datasets.CIFAR10(root=path, train=True, download=True)
    x = train_set.data
    y = train_set.targets
    if subsample and len(y) > subsample:
        x, _, y, _ = train_test_split(x, y, train_size=subsample, stratify=y, random_stte=42)
    return x, y, 'image'

def load_cifar10_from_keras(path: str, subsample: int = 2000) -> Tuple[Any, Any, str]:
    """ Carrega o dataset CIFAR-10 usando a API do Keras. """
    # O Keras fará o download e cache dos dados automaticamente
    print("  - Carregando CIFAR-10 via TensorFlow/Keras...")
    from tensorflow.keras.datasets import cifar10

    (x, y), (_, _) = cifar10.load_data()

    # O Keras retorna os rótulos num formato (n_samples, 1), precisamos achatá-lo para (n_samples,)
    y = y.flatten()

    if subsample and len(y) > subsample:
        # Usamos train_test_split para fazer uma amostragem estratificada
        from sklearn.model_selection import train_test_split
        x, _, y, _ = train_test_split(x, y, train_size=subsample, stratify=y, random_state=42)

    return x, y, 'image'

def load_sms_spam(path: str, subsample: int = 4000) -> Tuple[Any, Any, str]:
    caminho_completo = os.path.join(path, "spam.csv")
    print(f"  - Carregando SMS Spam de: {caminho_completo}")

    try:
        df = pd.read_csv(
            caminho_completo,
            encoding='latin-1',
            sep=',',
            skiprows=202,
            header=None,
            names=['label', 'message'],
            engine='python',
            quoting=csv.QUOTE_NONE
        )

        # --- INÍCIO DA DEPURAÇÃO ---
        print(f"\n[DEBUG 1] Shape do DataFrame logo após a leitura: {df.shape}")
        print(f"[DEBUG 2] Valores únicos na coluna 'label' ANTES do map: {df['label'].unique()}\n")

        # Para evitar problemas com espaços ou maiúsculas/minúsculas, limpamos a coluna 'label'
        df['label'] = df['label'].str.strip().str.lower()

        # Mapeia os rótulos para valores numéricos
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})

        print(f"[DEBUG 3] Valores únicos na coluna 'label' DEPOIS do map: {df['label'].unique()}")
        print(f"[DEBUG 4] Contagem de valores nulos (NaN) na coluna 'label': {df['label'].isnull().sum()}\n")

        # Remove linhas onde o mapeamento falhou ou onde a mensagem está vazia
        df.dropna(subset=['label', 'message'], inplace=True)

        print(f"[DEBUG 5] Shape do DataFrame APÓS remover os NaNs: {df.shape}\n")
        # --- FIM DA DEPURAÇÃO ---

        if df.empty:
            print("ERRO CRÍTICO: O DataFrame ficou vazio após a limpeza. Verifique os valores em 'label'.")
            return None, None, None

        df['label'] = df['label'].astype(int)

        x_df = df.drop('label', axis=1)
        y = df['label'].values

        if subsample and len(y) > subsample:
            from sklearn.model_selection import train_test_split
            x_df, _, y, _ = train_test_split(x_df, y, train_size=subsample, stratify=y, random_state=42)

        return x_df['message'].tolist(), y, 'text'

    except Exception as e:
        print(f"  ERRO DETALHADO ao ler ou processar o CSV: {e}")
        return None, None, None

def load_bank_marketing(path: str, subsample: int = 4000) -> Tuple[Any, Any, str]:
    df = pd.read_csv(os.path.join(path, "bank-full.csv"), sep=';')
    df['y'] = df['y'].map({'no': 0, 'yes': 1})
    x = df.drop('y', axis=1)
    y = df['y'].values
    if subsample and len(y) > subsample:
        x, _, y, _ = train_test_split(x, y, train_size=subsample, stratify=y, random_state=42)
    return x, y, 'numeric'

def load_hate_speech(path: str, subsample: int = 4000) -> Tuple[Any, Any, str]:
    caminho_completo = os.path.join(path, "labeled_data.csv")
    print(f"  - Carregando Hate Speech de: {caminho_completo}")
    try:
        df = pd.read_csv(os.path.join(path, "labeled_data.csv"), sep=',', on_bad_lines='skip')
        df.columns = df.columns.str.replace(';', '').str.strip()
    except FileNotFoundError:
        print(f"AVISO: Arquivo não encontrado em '{caminho_completo}'. Pulando.")
        return None, None, None
    except Exception as e:
        print(f"  ERRO ao ler o arquivo CSV: {e}")
        return None, None, None

    if 'tweet' not in df.columns or 'class' not in df.columns:
        print("  ERRO: O CSV do Hate Speech não contém as colunas 'tweet' e 'class'.")
        return None, None, None

    df.dropna(subset=['tweet', 'class'], inplace=True)
    x = df['tweet'].tolist()
    y = df['class'].values

    if subsample and len(y) > subsample:
        # Conta quantos exemplos existem em cada classe
        class_counts = pd.Series(y).value_counts()

        # VErifica se alguma classe tem menos de 2 membros
        if (class_counts < 2).any():
            print(" AVISO: Pelo menos uma classe tem menos 2 membros. ")
            indices = np.random.choice(len(y), size=subsample, replace=False)
            x_sub = [x[i] for i in indices]
            y_sub = y[indices]
            return x_sub, y_sub, 'text'
        else:
            # Se todas as classes são seguras, usa a estratificação (metodo preferencial)
            x_sub, _, y_sub, _ = train_test_split(x, y, train_size=subsample, stratify=y, random_state=42)
            return x_sub, y_sub, 'text'
        #x, _, y, _ = train_test_split(x, y, train_size=subsample, stratify=y, random_state=42)

    return x, y, 'text'

def load_sms_spam1(path: str, subsample: int = 4000) -> Tuple[Any, Any, str]:
    caminho_completo = os.path.join(path, "spam.csv")
    print(f"  - Carregando SMS Spam de: {caminho_completo}")

    try:
        # --- LEITURA PRECISA USANDO PANDAS ---
        # Esta combinação de parâmetros é a receita para ler o seu arquivo específico.
        df = pd.read_csv(
            caminho_completo,
            encoding='latin-1',  # Para os caracteres especiais
            sep=';',  # O separador correto entre 'label' e 'message'
            header=0,  # Usa a primeira linha que encontrar como cabeçalho
            usecols=[0, 1]  # Carrega APENAS as duas primeiras colunas e ignora o lixo (;;;)
        )
        print(df.head())

        # O pandas nomeará as colunas como 'v1' e 'v2' com base no cabeçalho lido.
        # Vamos renomeá-las para nomes mais claros.
        if 'v1' in df.columns and 'v2' in df.columns:
            df.rename(columns={'v1': 'label', 'v2': 'message'}, inplace=True)
        else:
            # Plano B se os nomes forem diferentes
            df.columns = ['label', 'message']

        # O resto do código para limpar e preparar os dados
        df.dropna(subset=['label', 'message'], inplace=True)
        df['label'] = df['label'].str.strip().str.lower()
        df['label'] = df['label'].map({'ham': 0, 'spam': 1})
        df.dropna(subset=['label'], inplace=True)
        df['label'] = df['label'].astype(int)

        x_df = df.drop('label', axis=1)
        y = df['label'].values

        if subsample and len(y) > subsample:
            from sklearn.model_selection import train_test_split
            x_df, _, y, _ = train_test_split(x_df, y, train_size=subsample, stratify=y, random_state=42)

        return x_df['message'].tolist(), y, 'text'

    except Exception as e:
        print(f"  ERRO DETALHADO ao ler ou processar o CSV: {e}")
        return None, None, None


if __name__ == "__main__":
    BASE_DATA_PATH = "datasets_locais"

    PLOTS_DIR = "plots"
    os.makedirs(PLOTS_DIR, exist_ok=True)

    # MApeia nome do dataset -> (função_de_carregamento, kwargs, tipo_de_dado)
    dataset_registry = {
        # "CIFAR-10": (load_cifar10_from_keras, {'subsample': 1000}),
        #"BANK MARKETING": (load_bank_marketing, {'subsample': 1000}),
        "SMS SPAM": (load_sms_spam1, {'subsample': 1000}),
        #"HATE SPEECH": (load_hate_speech, {'subsample': 1000})
    }
    processor = DataProcessor()
    reducer = DimensionalityReducer(n_components=2)
    evaluator = ReductionEvaluator()

    methods_to_run = ['pca', 'tsne', 'umap', 'mds']
    #methods_to_run = ['pca', 'tsne', 'mds']
    # methods_to_run = ['pca', 'tsne', 'mds', 'umap']

    for name, (loader_fn, kwargs) in dataset_registry.items():
        dataset_path = os.path.join(BASE_DATA_PATH, name.lower().replace(' ', '_').replace('-', '_'))
        print(f"\n{'=' * 25} PROCESSANDO DATASET: {name.upper()} {'=' * 25}")

        try:
            x_raw, y, data_type = loader_fn(dataset_path, **kwargs)
        except FileNotFoundError:
            print(f"AVIDSO: Arquivos para o  dataset {name} não encontrados em '{dataset_path}'. Pulando.")
            continue
        except Exception as e:
            print(f"ERRO ao carregar o dataset '{name}': {e}. Pulando este dataset")
            continue

        if data_type == 'numeric':
            x_processed = processor.preprocess_numeric(x_raw)
        elif data_type == 'text':
            x_processed = processor.preprocess_text(x_raw)
        elif data_type == 'image':
            x_processed = processor.preprocess_image(x_raw)

        for method in methods_to_run:
            try:
                print(f"\n-> Aplicando {method.upper()} em '{name}'...")
                start_time = time.time()
                x_reduced = reducer.reduce(x_processed, method)
                exec_time = time.time() - start_time

                evaluator.evaluate(x_processed, x_reduced, name, method, exec_time)
                evaluator.plot_results(x_reduced, name, method, PLOTS_DIR, y)
            except Exception as e:
                print(f"ERRO ao excutar {method.upper()} em '{name}': {e}.")

        results_df = evaluator.get_results_dataframe()
        if not results_df.empty:
            print("\n\n" + "=" * 70)
            print(" - RESULTADOPS FINAIS DAS METRICAS")
            print("=" * 70)
            print(results_df.round(4).to_string())

            # =============================================================
            # CÓDIGO PARA SALVAR OS RESULTADOS EM ARQUIVO
            # =============================================================

            nome_arquivo_csv = "resultados_metricas.csv"
            arquivo_existe = os.path.exists(nome_arquivo_csv)
            results_df.to_csv(nome_arquivo_csv, mode='a', header=not arquivo_existe, index=False)

            print(f"\n[SUCESSO] Resultados das métricas foram salvos em: {nome_arquivo_csv}")
