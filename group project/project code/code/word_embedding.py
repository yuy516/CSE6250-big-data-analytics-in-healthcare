from statistics import mean

import nltk
from gensim.models import KeyedVectors
from keras_preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd
import numpy as np

import re

from sklearn.decomposition import PCA

from keras.preprocessing.text import Tokenizer

def read_csv(filepath, catagory):
    df_train = pd.read_csv(filepath + catagory + '/' + catagory + '_train.csv')
    df_train = df_train.drop(columns=df_train.columns[0])

    df_test = pd.read_csv(filepath + catagory + '/' + catagory + '_test.csv')
    df_test = df_test.drop(columns=df_test.columns[0])

    df_all = pd.read_csv(filepath + catagory + '/' + catagory + '.csv')
    df_all = df_all.drop(columns=df_all.columns[0])

    return df_train, df_test, df_all

def get_list(data):
    text = []
    label = []
    for i, row in data.iterrows():
        text.append(row['text'])
        label.append(row['label'])
    return text, label

def preprocess_text(text):
    # lower casing
    text = np.char.lower(text)

    # Punctuation and numeric values removal
    text = np.char.replace(text, "'", "")
    symbols = "|!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in symbols:
        text = np.char.replace(text, i, ' ')
    # processed_text = list(map(lambda x: re.sub(r'[^A-Za-z ]', '', x), text))
    processed_text = list(filter(lambda x: x, map(lambda x: re.sub(r'[^A-Za-z ]', '', x), text)))
    for i in range(len(processed_text)):
        processed_text[i] = ' '.join(processed_text[i].split())

    # Lemmatization of text
    processed_text = lemmatization(processed_text)

    # filter non english word
    processed_text = filter_words(processed_text)

    # Remove stop words
    processed_text = remove_stopwords(processed_text)

    return processed_text

def embedding_word(text, label, method):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(text)

    word_map = tokenizer.word_index
    word_map_reverse = {k: v for v, k in word_map.items()}

    x_train_sequences = tokenizer.texts_to_sequences(text)

    if method == 'word2vec':
        # Word2Vec Using Gensim
        word_vectors = KeyedVectors.load_word2vec_format('../lib/word2vec-GoogleNews-vectors-negative300.bin',
                                                     binary=True)
    elif method == 'fasttext':
        # Fast Text Using Gensim
        word_vectors = KeyedVectors.load_word2vec_format('../lib/fasttext-300d-2M.vec', binary=False)
    elif method == 'glove':
        word_vectors = KeyedVectors.load_word2vec_format('../lib/glove.6B.300d.w2vformat.txt', binary=False)

    X = []
    for row in x_train_sequences:
        x = []
        for n in row:
            word = word_map_reverse[n]
            if word in word_vectors:
                embedding_vector = word_vectors[word]
                for i in range(300):
                    x.append(embedding_vector[i])
        X.append(x)

    length_train_texts = [len(x) for x in X]
    max_length = int(mean(length_train_texts)) + int(np.std(length_train_texts))
    padded_train_data = pad_sequences(X, maxlen=max_length)
    pca_model = PCA(n_components=min(len(text), max_length//300))
    pca_model.fit(padded_train_data)
    X_comps = pca_model.transform(padded_train_data)
    Y = np.array(label).reshape(len(X_comps), 1)
    data = np.append(X_comps, Y, 1)
    return data

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    new = []

    for str in text:
        word_tokens = word_tokenize(str)
        processed_data = [lemmatizer.lemmatize(word) for word in word_tokens]
        new.append(' '.join(processed_data))
    return new

def filter_words(text):
    words = set(nltk.corpus.words.words())
    new = []

    for str in text:
        word_tokens = word_tokenize(str)
        processed_data = [word for word in word_tokens if word in words]
        new.append(' '.join(processed_data))
    return new

def remove_stopwords(text):
    new=[]
    stop_words = set(stopwords.words("english"))
    for str in text:
        word_tokens = word_tokenize(str)
        filtered_text = [word for word in word_tokens if word not in stop_words]
        new.append(' '.join(filtered_text))
    return new

def main():
    filepath = '../data/processed/'
    diseases = ['Asthma', 'CAD', 'CHF', 'Depression', 'Diabetes', 'Gallstones', 'GERD', 'Gout', 'HC', 'Hypertension',
                'HT', 'OA', 'Obesity', 'OSA', 'PVD', 'VI']
    method = input('Word embedding method:')
    if method == "":
        method = "word2vec"
    print("---------------------------")
    print('Word embedding method:' + method)
    print("---------------------------")

    for disease in diseases:
        df_train, df_test, df_all = read_csv(filepath, disease)
        text, label = get_list(df_all)

        new_text = preprocess_text(text)
        data = embedding_word(new_text, label, method)

        data_df = pd.DataFrame(data)
        outpath = '../data/'+method+'/'+disease+'/data.csv'
        data_df.to_csv(outpath)
        print('Data for {} saved!'.format(disease))

if __name__ == "__main__":
    main()