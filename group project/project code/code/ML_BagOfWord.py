import time
from statistics import mean

import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

import pandas as pd
import numpy as np

import re

from sklearn.feature_selection import SelectFromModel, SelectKBest, f_classif
from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

import warnings

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

    X = np.array(processed_text)
    Y = np.array(label)

    Tfidf_vect = TfidfVectorizer()
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

def svm_pred(X_train, Y_train, X_test):
    model = SVC()
    classifier = model.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    return Y_pred

def naive_bayes_pred(X_train, Y_train, X_test):
    model = GaussianNB()
    classifier = model.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    return Y_pred

def random_forest_pred(X_train, Y_train, X_test):
    model = RandomForestClassifier()
    classifier = model.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    return Y_pred

def knn1_pred(X_train, Y_train, X_test):
    model = KNeighborsClassifier(n_neighbors=1)
    classifier = model.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    return Y_pred

def knn5_pred(X_train, Y_train, X_test):
    model = KNeighborsClassifier(n_neighbors=5)
    classifier = model.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    return Y_pred

def decision_tree_pred(X_train, Y_train, X_test):
    model = DecisionTreeClassifier(random_state=545510477, max_depth=5)
    classifier = model.fit(X_train, Y_train)
    Y_pred = classifier.predict(X_test)
    return Y_pred

def classification_metrics(Y_pred, Y_true):
    acc = accuracy_score(Y_true, Y_pred)
    fpr, tpr, thresholds = roc_curve(Y_true, Y_pred)
    auc_ = auc(fpr, tpr)
    precision = precision_score(Y_true, Y_pred)
    recall = recall_score(Y_true, Y_pred)
    f1 = f1_score(Y_true, Y_pred, zero_division=1)
    return acc, auc_, precision, recall, f1

def get_matrics_kfold(X, Y, k=10, model=svm_pred):
    kf = KFold(n_splits=k, shuffle=True)
    acc_t = []
    auc_t = []
    f1_t = []

    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        Y_train, Y_test = Y[train_index], Y[test_index]
        Y_pred = model(X_train, Y_train, X_test)
        new_y_test, new_y_pred = swap_label(Y_test, Y_pred)

        old_acc, old_auc, old_precision, old_recall, old_f1 = classification_metrics(Y_pred, Y_test)
        new_acc, new_auc, new_precision, new_recall, new_f1 = classification_metrics(new_y_pred, new_y_test)

        acc_t.append(mean([old_acc,new_acc]))
        auc_t.append(mean([old_auc,new_auc]))
        f1_t.append(mean([old_f1, new_f1]))

    acc_avg = mean(acc_t)
    auc_avg = mean(auc_t)
    f1_avg = mean(f1_t)
    return acc_avg, auc_avg, f1_avg

def swap_label(Y_test, Y_pred):
    new_y_test = []
    new_y_pred = []
    for y in Y_test:
        if (y == 0):
            new_y_test += [1]
        else:
            new_y_test += [0]
    for y in Y_pred:
        if (y == 0):
            new_y_pred += [1]
        else:
            new_y_pred += [0]
    return new_y_test, new_y_pred

def ExtraTrees_feature_selection(X, Y):
    clf = ExtraTreesClassifier(n_estimators=100)
    clf = clf.fit(X, Y)
    model = SelectFromModel(clf, prefit=True)
    X_new = model.transform(X)
    return X_new

def solve_imbalance_data(data):
    df0 = data.loc[data['label'] == 0]
    n0 = len(df0)
    df1 = data.loc[data['label'] == 1]
    n1 = len(df1)
    newdata = data
    if (n0 > n1):
        df1 = pd.DataFrame(np.repeat(df1.values, n0//n1, axis=0), columns=['id', 'text', 'label'])
        newdata = pd.concat([df0, df1], ignore_index=True)
    return newdata

def main():
    warnings.filterwarnings("ignore")
    # nltk.download('punkt')
    # nltk.download('wordnet')
    # nltk.download('words')
    # nltk.download('stopwords')
    filepath = '../data/processed/'
    disease = input("Enter disease:")
    feature_selection = input("Enter feature selection type:")
    if disease == "":
        disease = 'Asthma'
    if feature_selection == "":
        feature_selection = 'All Features'
    print("---------------------------")
    print('Disease Name:' + disease)
    print('Feature Selection Type:'+ feature_selection)
    df_train, df_test, df_all = read_csv(filepath, disease)
    df_all = solve_imbalance_data(df_all)
    text, label = get_list(df_all)
    X, Y = preprocess_text(text, label)
    if feature_selection == 'ExtraTreesClassifier':
        X = ExtraTrees_feature_selection(X, Y)
    elif feature_selection == 'SelectKBest':
        X = SelectKBest(f_classif, k=X.shape[1]//5).fit_transform(X, Y)

    print('The shape of TF-IDF matrix is :', X.shape)
    models = [('svm', svm_pred),
              ('naive bayes', naive_bayes_pred),
              ('random forest', random_forest_pred),
              ('knn1', knn1_pred),
              ('knn5', knn5_pred),
              ('decision tree', decision_tree_pred)]
    for model_name, model in models:
        start_time = time.time()
        acc_avg, auc_avg, f1_avg = get_matrics_kfold(X, Y, 10, model)
        print('-----The Metrics Report------')
        print('>>> Model Name:' + model_name)
        print('>>> Training Time:', time.time() - start_time)
        print('>>> Accuracy:', acc_avg)
        print('>>> AUC:', auc_avg)
        print('>>> F1:', f1_avg)

if __name__ == "__main__":
    main()