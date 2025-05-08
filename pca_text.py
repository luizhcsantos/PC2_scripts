
import nltk
import requests
import zipfile
import os
from io import BytesIO
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
import numpy as np
from PIL import Image

#!pip install ucimlrepo
#from ucimlrepo import fetch_ucirepo

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split

import time 
from sklearn.metrics import pairwise_distances
from sklearn.manifold import trustworthiness
from sklearn.neighbors import NearestNeighbors

from sklearn.feature_extraction.text import CountVectorizer
import chardet 
import string
from nltk.corpus import stopwords
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')
nltk.download('punkt_tab')
from sklearn.feature_extraction.text import TfidfTransformer

import re
from wordcloud import WordCloud
from mlxtend.frequent_patterns import fpgrowth
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

class LemmaTokenizer(object):

    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.stopwords = set(stopwords.words('english'))

    def __call__(self, document):
        lemmas = []
        
        # Pre-proccessing of one document at the time
        # Removing puntuation
        translator_1 = str.maketrans(string.punctuation, ' ' *
                                     len(string.punctuation))
        document = document.translate(translator_1)

        # Removing numbers
        document = re.sub(r'\d+', ' ', document)

        # Removing special characters
        document = re.sub(r"[^a-zA-Z0-9]+", ' ', document)

        # The document is a string up to now, after word_tokenize(document) we'll work on every word one at the time
        for token in word_tokenize(document):
            
            # Removing spaces
            token = token.strip()
            
            # Lemmatizing
            token = self.lemmatizer.lemmatize(token)

            # Removing stopwords
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

    # Load the dataset

    sms = pd.read_csv("SMS/spam.csv", encoding='latin-1')
    sms.dropna(how="any", inplace=True, axis=1)
    sms.columns = ['label', 'message']

    sms['label_num'] = sms.label.map({'ham':0, 'spam':1})
    sms['message_len'] = sms.message.apply(len)

    sms['clean_msg'] = sms.message.apply(text_process)



    vectorizer = TfidfVectorizer(tokenizer=LemmaTokenizer())

    X_tfidf = vectorizer.fit_transform(sms.clean_msg)

    pca = PCA(n_components=0.90, random_state=0)
    x_pca = pca.fit_transform(X_tfidf.toarray())

    # svd = TruncatedSVD(n_components=100, random_state=42)
    # X_svd = svd.fit_transform(X_tfidf)


    """X = sms.clean_msg
    y = sms.label_num


    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    vect = CountVectorizer(stop_words='english', max_df=0.5)
    vect.fit(X_train)

    X_train_dtm = vect.transform(X_train)

    X_train_dtm = vect.fit_transform(X_train)

    X_test_dtm = vect.transform(X_test)

    tfidf_transformer = TfidfTransformer()
    tfidf_transformer.fit(X_train_dtm)
    tfidf_transformer.transform(X_train_dtm)

    pca = PCA(n_components=0.90, random_state=0)
    principal_components = pca.fit_transform(X_train_dtm.toarray()) """
    


if __name__ == "__main__":
    main()
