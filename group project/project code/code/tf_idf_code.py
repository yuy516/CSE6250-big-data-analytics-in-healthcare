# -*- coding: utf-8 -*-
#Importing the packages and libraries 
#Gensim Library
import gensim
from gensim.models import Word2Vec 
from gensim.scripts.glove2word2vec import glove2word2vec 
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors
from gensim.utils import tokenize
#import multiprocessing

#NlTK Library
import nltk
import nltk.tokenize
from nltk.tokenize import sent_tokenize
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer 

#Pandas and Numpy
import pandas as pd
import numpy as np 
from numpy import array
from numpy import asarray
from numpy import zeros
import statistics 
from statistics import mean

#Keras
import keras 
from keras.layers import Embedding
from keras.models import Sequential
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, SimpleRNN
from keras.metrics import binary_accuracy
from keras.layers import Dense, Flatten, Dropout, Activation, Embedding, LSTM, Bidirectional, SimpleRNN, Conv1D, MaxPooling1D, TimeDistributed

#Sci-Kit Library 
import sklearn
from sklearn import metrics, svm
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_squared_error, mean_absolute_error

#Miscellaneous 
import argparse
import os
import io
import re
import sys
import gc
import string
import pickle
import datetime
import matplotlib.pyplot as plt
import sys
import gc
import statistics
from statistics import mean

# In[14]:

if len(sys.argv)!=2:
    print ("python test.py classname")
    sys.exit(1)
target_class = sys.argv[1]
print("---------------------------")
print ("CLASSNAME:",target_class)
train_data = pd.read_csv('../data/processed/Asthma/Asthma.csv')
for column in train_data:
    print(column)
    
#Defining the function for seperating Texts and Labels
def texts_and_labels(data):
    texts = []
    labels = []
    #id = []
    for i,r in data.iterrows():
        texts += [r['text']]
        labels += [r['label']]
        #id += [r['id']]
    return texts, labels #, id

#Getting the Text and Labels of Input Data
text, label = texts_and_labels(train_data)
print('Labels distribution:', label.count(0), label.count(1))
print(type(text), len(text))
print(label[0:50], 'The Label type is', type(label), len(label))

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
#Pre-processing the data
#Lower casing the data
text1=np.char.lower(text)
print(type(text1))   
text2= np.char.replace(text1, "'", "")
temp_text = text2
#Defining the symbols
symbols = "|!\"#$%&()*+-./:;<=>?@[\]^_`{|}~\n"
for i in symbols:
    temp_text = np.char.replace(temp_text,i,' ')
text3 = temp_text
#Removing stopwords
nltk.download('stopwords')
nltk.download('punkt')
def remove_stopwords(text): 
    new=[]
    stop_words = set(stopwords.words("english")) 
    for str in text:
        word_tokens = word_tokenize(str) 
        filtered_text = [word for word in word_tokens if word not in stop_words]
        new.append(' '.join(filtered_text))
    return new
text = remove_stopwords(text3)
processed_data=list(filter(lambda x:x, map(lambda x:re.sub(r'[^A-Za-z ]', '', x), text)))
for i in range(0, len(processed_data)):
    processed_data[i] = ' '.join(processed_data[i].split())
print(type(processed_data))

#Calculating the length and average length of text
sum=[]
viv=0
for l in processed_data:
    viv += len(l)
    sum.append(len(l))
print('Type of sum is', type(sum))
avg=int((viv/len(processed_data)))
print(avg)
print ("Max length is :", max(sum))  
print ("MIN length is :", min(sum)) 
print('Std dev is:', np.std(sum))
max_len=max(sum)

# Preparing Input Data
X = np.array(processed_data)
Y= np.array(label)
print(Y[0:10])
print('The type X and Y are :', type(X), type(Y))
print('The shape X and Y are :', X.shape, Y.shape)

#Using the TFiDF vectorizer to vectorize the data
Tfidf_vect = TfidfVectorizer(max_features=max_len)
X = Tfidf_vect.fit_transform(X).toarray() 
print('The type of TF-IDF matrix and Shape is :', type(X), X.shape) 
acc = []
p = []
r = []
f = []
ba = []
results = []

#Defining the models:
# 1. SVM Model
model1 = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
# 2. RandomForest Classifier
model2 = RandomForestClassifier(max_depth=None, random_state=None)
# 3. Naive Bayes Model
model3 = GaussianNB()

kf = KFold(n_splits=10, shuffle=False)
kf.get_n_splits(X)
for train_index, test_index in kf.split(X):
    print('',train_index[0:5], type(train_index))
    print(test_index[0:5], type(test_index))
    x_train_text, x_test_text=X[train_index], X[test_index] 
    y_train_label, y_test_label=Y[train_index], Y[test_index] 
    print('The shape of x_train_text and x_test_text are:', x_train_text.shape, x_test_text.shape)
    print('The type of x_train_text and x_test_text are:', type(x_train_text), type(x_test_text))
    print('The shape of y_train_label and y_test_label are:', y_train_label.shape, y_test_label.shape)
    print('The type of y_train_label and y_test_label are:', type(y_train_label), type(y_test_label))

    print('Old evaluation:')
    model = model1.fit(x_train_text, y_train_label)
    pred_labels=model.predict(x_test_text)
    print('\nOriginal classes:', y_test_label[:20], '\n', len(y_test_label))
    print('Predicted classes', pred_labels[:10], '\n', len(pred_labels), type(pred_labels))
    print('-----The 1st Classification Report')
    print(classification_report(y_test_label, pred_labels, digits=4))
    print('-----The 1st Confusion Matrix')
    print('The confusion matrix is', '\n', confusion_matrix(y_test_label, pred_labels))

    #Generating a CSV File of predicted results
    pred=pd.DataFrame(columns=['Id', 'Orginal Labels', target_class])
    pred['Id'] = test_index
    pred['Orginal Labels'] = y_test_label
    pred[target_class] = pred_labels
    
    print('The data Frame pred results ', pred[:5])
    results += [pred]  
    # Computing the First metrics:
    acc_binary = accuracy_score(y_test_label, pred_labels)
    p_binary = precision_score(y_test_label, pred_labels)
    r_binary = recall_score(y_test_label, pred_labels)
    f_binary = f1_score(y_test_label, pred_labels)
    b_acc = balanced_accuracy_score(y_test_label, pred_labels)
    
    print('-----The 1st Metrics Report------')
    print('>>> Accuracy:', acc_binary)
    print('>>> Precision:', p_binary)
    print('>>> Recall:', r_binary)
    print('>>> F1:', f_binary)
    print('>>> Balanced Accuracy:', b_acc)

    #Swapping the 0 an 1 of the test and predicted labels
    print('new method2')
    new_y_test_label = []
    new_pred_labels = []
    for each_value_1 in y_test_label:
        if(each_value_1 == 0):
            new_y_test_label += [1]
        else:
            new_y_test_label += [0]   
    for each_value_1 in pred_labels:
        if(each_value_1 == 0):
            new_pred_labels += [1]
        else:
            new_pred_labels += [0]

    print('new_y_test_label:', new_y_test_label[:10], '\n', type(new_y_test_label), len(new_y_test_label))
    print('new_pred_labels:', new_pred_labels[:10], '\n', type(new_pred_labels), len(new_pred_labels))
    
    print('-----The 2nd Classification Report')
    print(classification_report(new_y_test_label, new_pred_labels, digits=4))

    print('-----The 2nd Confusion Matrix')
    print('The confusion matrix is', '\n', confusion_matrix(new_y_test_label, new_pred_labels))

    #Computing the new metrics:
    new_acc_binary = accuracy_score(new_y_test_label, new_pred_labels)
    new_p_binary = precision_score(new_y_test_label, new_pred_labels)
    new_r_binary = recall_score(new_y_test_label, new_pred_labels)
    new_f_binary = f1_score(new_y_test_label, new_pred_labels)
    new_b_acc = balanced_accuracy_score(new_y_test_label, new_pred_labels)

    print('-----The 2nd Metrics Report------')
    print('>>> Accuracy:', new_acc_binary)
    print('>>> Precision:', new_p_binary)
    print('>>> Recall:', new_r_binary)
    print('>>> F1:', new_f_binary)
    print('>>> Balanced Accuracy:', new_b_acc)

    print('Caluclating the mean of the both metrics')
    acc_binary = (acc_binary+new_acc_binary)/2
    p_binary = (p_binary+new_p_binary)/2
    r_binary = (r_binary+new_r_binary)/2
    f_binary = (f_binary+new_f_binary)/2
    b_acc = (b_acc+new_b_acc)/2
    
    #Adding the metrics values to their respective lists
    acc += [acc_binary]
    p += [p_binary]
    r += [r_binary]
    f += [f_binary]
    ba += [b_acc]

    print('-----The final Metrics Report------')
    print('>>> Accuracy:', acc_binary)
    print('>>> Precision:', p_binary)
    print('>>> Recall:', r_binary)
    print('>>> F1:', f_binary)
    print('>>> Balanced Accuracy:', b_acc)

    del model
    print('del model')
    model = None
    print('no model')
    model=model1
    print('new model loaded')
    
print('---- The final Averaged result after 10-fold validation: ' , target_class)
print('>> Accuracy:', mean(acc)*100)
print('>> Precision:', mean(p)*100)
print('>> Recall:', mean(r)*100)
print('>> F1:', mean(f)*100)
print('>> Balanced Accuracy:', mean(ba)*100)
pred_results = pd.concat(results, axis=0, join='inner').sort_index()
print(pred_results [0:100])   

