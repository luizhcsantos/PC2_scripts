
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
from ucimlrepo import fetch_ucirepo

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
from sklearn.feature_extraction.text import TfidfTransformer



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

    X = sms.clean_msg
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
    principal_components = pca.fit_transform(X_train_dtm.toarray())
    


if __name__ == "__main__":
    main()
