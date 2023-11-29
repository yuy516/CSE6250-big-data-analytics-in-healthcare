import time
from statistics import mean

import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, roc_curve, auc, precision_score, recall_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import KFold
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

def read_csv(filepath):
    df_all = pd.read_csv(filepath)
    df_all = df_all.drop(columns=df_all.columns[0])

    return df_all

def get_list(data):
    text = []
    label = []
    for i, row in data.iterrows():
        text.append(row.iloc[:-1])
        label.append(row.iloc[-1])
    X = np.array(text)
    Y = np.array(label)
    return X, Y

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

def solve_imbalance_data(data):
    df0 = data.loc[data.iloc[:, -1] == 0]
    n0 = len(df0)
    df1 = data.loc[data.iloc[:, -1] == 1]
    n1 = len(df1)
    newdata = data

    if (n0 > n1):
        df1 = pd.DataFrame(np.repeat(df1.values, n0//n1, axis=0))
        newdata = pd.concat([df0, df1], ignore_index=True)
    elif (n1 > n0):
        df0 = pd.DataFrame(np.repeat(df0.values, n1 // n0, axis=0))
        newdata = pd.concat([df0, df1], ignore_index=True)
    newdata.fillna(0, inplace = True)
    return newdata

def main():
    # warnings.filterwarnings("ignore")

    # nltk.download('punkt')
    # nltk.download('wordnet')
    # nltk.download('words')
    # nltk.download('stopwords')

    method = input('Word embedding method:')
    models = [('svm', svm_pred, []),
              ('naive bayes', naive_bayes_pred, []),
              ('random forest', random_forest_pred, []),
              ('knn1', knn1_pred, []),
              ('knn5', knn5_pred, []),
              ('decision tree', decision_tree_pred, [])]
    diseases = ['Asthma', 'CAD', 'CHF', 'Depression', 'Diabetes', 'Gallstones', 'GERD', 'Gout', 'HC', 'Hypertension',
                'HT', 'OA', 'Obesity', 'OSA', 'PVD', 'VI']
    if method == "":
        method = "word2vec"
    print("---------------------------")
    print('Word embedding method:' + method)

    for disease in diseases:
        print("---------------------------")
        print('Disease Name:' + disease)
        filepath = '../data/'+method+'/'+disease+'/data.csv'
        df_all = read_csv(filepath)
        df_all = solve_imbalance_data(df_all)
        X, Y = get_list(df_all)

        for model_name, model, avg in models:
            start_time = time.time()
            acc_avg, auc_avg, f1_avg = get_matrics_kfold(X, Y, 10, model)
            avg.append([acc_avg, auc_avg, f1_avg])
            print('-----The Metrics Report------')
            print('>>> Model Name:' + model_name)
            print('>>> Training Time:', time.time() - start_time)
            print('>>> Accuracy:', acc_avg)
            print('>>> AUC:', auc_avg)
            print('>>> F1:', f1_avg)

    print('-----The Final Report------')
    for model_name, model, avg in models:
        print("---------------------------")
        average = np.array(avg)
        column_mean = np.mean(average, axis=0)
        print('>>> Model Name:' + model_name)
        print('>>> Average Accuracy:', column_mean[0])
        print('>>> Average AUC:', column_mean[1])
        print('>>> Average F1:', column_mean[2])

if __name__ == "__main__":
    main()