from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

import pandas as pd
import numpy as np

import re

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def read_csv(filepath, catagory):
    df_train = pd.read_csv(filepath + catagory + '/' + catagory + '_train.csv')
    df_train = df_train.drop(columns=df_train.columns[0])

    df_test = pd.read_csv(filepath + catagory + '/' + catagory + '_test.csv')
    df_test = df_test.drop(columns=df_test.columns[0])

    df_all = pd.read_csv(filepath + catagory + '/' + catagory + '.csv')
    df_all = df_all.drop(columns=df_all.columns[0])

    return df_train, df_test, df_all

# get text and label as list
def get_list(data):
    text = []
    label = []
    for i, row in data.iterrows():
        text.append(row['text'])
        label.append(row['label'])
    return text, label

def preprocess_text(text, label):
    # lower casing
    text = np.char.lower(text)

    # Punctuation and numeric values removal
    text = np.char.replace(text, "'", "")
    symbols = "|!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
    for i in symbols:
        text = np.char.replace(text, i, ' ')
    processed_text = list(map(lambda x: re.sub(r'[^A-Za-z ]', '', x), text))
    for i in range(len(processed_text)):
        processed_text[i] = ' '.join(processed_text[i].split())

    # nltk.download('punkt')
    # nltk.download('wordnet')

    # Lemmatization of text
    processed_text = lemmatization(processed_text)

    # remove of suffix
    # processed_text = lemmatization(processed_data)

    X = np.array(processed_text)
    Y = np.array(label)

    Tfidf_vect = TfidfVectorizer(max_features=1000)
    X = Tfidf_vect.fit_transform(X).toarray()
    return X, Y

def lemmatization(text):
    lemmatizer = WordNetLemmatizer()
    new = []

    for str in text:
        word_tokens = word_tokenize(str)
        processed_data = [lemmatizer.lemmatize(word) for word in word_tokens]
        new.append(' '.join(processed_data))
    return new

def remove_suffix(text):
    stemmer = PorterStemmer()
    new = []

    for str in text:
        word_tokens = word_tokenize(str)
        processed_data = [stemmer.stem(word) for word in word_tokens]
        new.append(' '.join(processed_data))
    return new

def compare_models(X_train, Y_train):
    models = []
    models.append(('Naive Bayes', GaussianNB()))
    models.append(('SVM', SVC(C=1.0, kernel='linear', degree=3, gamma='auto')))
    models.append(('Random Forest', RandomForestClassifier(max_depth=None, random_state=None)))
    models.append(('KNN1', KNeighborsClassifier(n_neighbors=1)))
    models.append(('KNN5', KNeighborsClassifier(n_neighbors=5)))
    models.append(('Decision Tree', DecisionTreeClassifier(random_state=545510477, max_depth=5)))

    for name, model in models:
        kf = KFold(n_splits=10)
        res = cross_val_score(model, X_train, Y_train, cv=kf, scoring='f1')
        msg = "Average F1 score for %s: %f" % (name, res.mean())
        print(msg)

def main():
    filepath = '../data/processed/'
    df_train, df_test, df_all = read_csv(filepath, 'Asthma')
    text, label = get_list(df_all)
    X, Y = preprocess_text(text, label)
    print('The type of TF-IDF matrix is :', X.shape)
    compare_models(X, Y)

if __name__ == "__main__":
    main()