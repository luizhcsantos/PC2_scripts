import os
import string
import time
import nltk
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from sklearn.decomposition import PCA
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances
from sklearn.neighbors import NearestNeighbors
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
import re
import chardet
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer

class LemmaTokenizer(object):

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))

    def __call__(self, document):
        lemmas = []
        
        # Pre-processamento de um documento por vez
        # Remoção de pontuação
        translator_1 = str.maketrans(string.punctuation, ' ' *
                                     len(string.punctuation))
        document = document.translate(translator_1)

        # Remoção de numeros
        document = re.sub(r'\d+', ' ', document)

        # Remoção de caracteres especiais
        document = re.sub(r"[^a-zA-Z0-9]+", ' ', document)

        for token in word_tokenize(document):
            
            # Remoção de espaços
            token = token.strip()
            
            # Lematização
            token = self.lemmatizer.lemmatize(token)

            # Remoção de stopwords
            if token not in self.stopwords and len(token) > 2:
                lemmas.append(token)
        return lemmas



def text_process(mess):
    """
    Takes in a string of text, then performs the following:
    1. Remove all punctuation
    2. Remove all stopwords
    3. Returns a list of the cleaned text
    """
    STOPWORDS = stopwords.words('english') + ['u', 'ü', 'ur', '4', '2', 'im', 'dont', 'doin', 'ure']
    # Check characters to see if they are in punctuation
    nopunc = [char for char in mess if char not in string.punctuation]

    # Join the characters again to form the string.
    nopunc = ''.join(nopunc)
    
    # Now just remove any stopwords
    return ' '.join([word for word in nopunc.split() if word.lower() not in STOPWORDS])


def main():

    n_execucoes = 5  # Número de execuções do PCA
    k_neighbors = 5  # Número de vizinhos para trustworthiness/continuity
    results = []  # Lista para armazenar resultados

    #caminho_conjunto = 'SMS/spam.csv'
    caminho_conjunto = 'hatespeech/labeled_data.csv'
    # Load the dataset
    with open(caminho_conjunto, 'rb') as f:
        result = chardet.detect(f.read())
    df = pd.read_csv(caminho_conjunto, encoding=result['encoding'])

    df.dropna(how="any", inplace=True, axis=1)
    # df.columns = ['label', 'message']
    #
    # df['label_num'] = df.label.map({'ham':0, 'spam':1})
    # df['message_len'] = df.message.apply(len)
    # df['clean_msg'] = df.message.apply(text_process)

    df['clean_msg'] = df.tweet.apply(text_process)
    print(df.columns)

    for run in range(n_execucoes):

        # Iniciar medição de tempo
        start_time = time.time()


        vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer(), token_pattern=None, max_features=10000)

        x_tfidf = vectorizer.fit_transform(df.clean_msg)

        pca = PCA(n_components=0.90, random_state=0)
        x_pca = pca.fit_transform(x_tfidf.toarray())

        # Calcular tempo de execução
        execution_time = time.time() - start_time

        # Calcular Stress
        original_distances = pairwise_distances(x_tfidf, metric='euclidean')
        pca_distances = pairwise_distances(x_pca, metric='euclidean')

        stress_numerator = np.sum((original_distances - pca_distances) ** 2)
        stress_denominator = np.sum(original_distances ** 2)
        stress = np.sqrt(stress_numerator / stress_denominator)

        # Calcular Trustworthiness
        trust = trustworthiness(x_tfidf, x_pca, n_neighbors=k_neighbors)

        # Calcular Continuity
        nbrs_original = NearestNeighbors(n_neighbors=k_neighbors).fit(x_tfidf)
        _, indices_original = nbrs_original.kneighbors(x_tfidf)

        nbrs_pca = NearestNeighbors(n_neighbors=k_neighbors).fit(x_pca)
        _, indices_pca = nbrs_pca.kneighbors(x_pca)

        continuity = 0
        n = x_tfidf.shape[0]
        for i in range(n):
            common_neighbors = len(set(indices_original[i]).intersection(set(indices_pca[i])))
            continuity += common_neighbors / k_neighbors
        continuity /= n

        # Salvar resultados
        results.append({
            'conjunto' : caminho_conjunto.split('/')[-1].split('.')[0],
            'execucao': run + 1,
            'tempo_execucao': execution_time,
            'stress': stress,
            'trustworthiness': trust,
            'continuity': continuity
        })
    caminho_resultados = 'resultados/pca/metricas_pca.csv'
    df_results = pd.DataFrame(results)
    if not os.path.isfile(caminho_resultados):
        # Se o arquivo não existe na pasta, cria com cabeçalho
        df_results.to_csv(caminho_resultados, index=False)
    else:
        # Se o arquivo já existe na pasta, adiciona novas linhas sem repetir o cabeçalho
        df_results.to_csv(caminho_resultados, mode='a', header=False, index=False)

    print(df_results)
    


if __name__ == "__main__":
    main()
