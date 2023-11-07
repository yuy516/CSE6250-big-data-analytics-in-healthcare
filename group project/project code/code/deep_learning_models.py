#Gensim Library
import gensim
from gensim.models import Word2Vec 
from gensim.scripts.glove2word2vec import glove2word2vec 
from gensim.utils import simple_preprocess
from gensim.models.keyedvectors import KeyedVectors
from gensim.utils import tokenize

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
from keras.utils import to_categorical
from keras.preprocessing import sequence
from keras.preprocessing.text import Tokenizer 
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, SimpleRNN
from keras.metrics import binary_accuracy
#from keras_self_attention import SeqSelfAttention, SeqWeightedAttention
from keras.layers import Dense, Flatten, Dropout, Activation, Embedding, LSTM, Bidirectional, SimpleRNN, Conv1D, MaxPooling1D, TimeDistributed

#Sci-Kit Library 
import sklearn
from sklearn import metrics
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn import datasets, linear_model
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,classification_report
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score, mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

#Miscellaneous 
import argparse
import os
import io
import re
import sys
import gc
import pickle
import datetime
import tensorflow as tf
import mxnet as mx
from bert_embedding import BertEmbedding
from scipy.sparse import random as sparse_random
import bert

class morbidity                                                                                                                                                                     :
    def __init__(self, target_class):
        self.target_class = target_class
        self.train_data = pd.read_csv('file_path' + target_class + '.csv', sep=';')
        print(type(self.train_data), len(self.train_data), '\n', self.train_data.head(3))
        self.train_data = self.train_data.sample(frac=1)  
        print(type(self.train_data), len(self.train_data), '\n', self.train_data.head(1))
        self.attention=attention()
        self.train_texts = None
        self.train_labels = None
        self.train_encoded_doc = None
        self.tokenizer = Tokenizer()
        self.mean_length = None
        self.max_length = None
        self.vocab_size = None
        self.padded_train_data = None
        self.embedding_matrix = None
        self.model = None
        self.embedding_dim = None
        
    def texts_and_labels(self):
        texts = []
        labels = []
        for i,r in self.train_data.iterrows():
            texts += [r['Text'].strip().split('\n', 1)[1]]
            labels += [r['Label']]
        self.train_texts = texts
        self.train_labels = labels
        print('Details of Training Data Text:', '\n', type(self.train_texts), len(self.train_texts))
        print('Details of Training Data Labels:', '\n', type(self.train_labels), len(self.train_labels), '\n', self.train_labels[0:10])
        print('Labels distribution of Training Labels:', '\n', 'Zeros-', self.train_labels.count(0), 'Ones=' ,self.train_labels.count(1))
        
    def padded_encoded_text(self):
        # Tokenizing the Data 
        self.tokenizer.fit_on_texts(self.train_texts) 
        # Defining the length of vocabulary
        self.vocab_size = len(self.tokenizer.word_index) + 1
        # Defining the vocabulary made from unique words
        self.my_vocab = set([w for (w,i) in self.tokenizer.word_index.items()])
        print('My Vocab set version is :', '\n', type(self.my_vocab), len(self.my_vocab))
        #Encoding the data to integar
        self.train_encoded_doc = self.tokenizer.texts_to_sequences(self.train_texts)
        print(type(self.train_encoded_doc), len(self.train_encoded_doc)) #, '\n', self.train_encoded_doc[0:5])
        # Calculating the average, standard deviation & maximum length of Encoded Training Data
        length_train_texts = [len(x) for x in self.train_encoded_doc]
        print ("Max length is :", max(length_train_texts))  
        print ("AVG length is :", mean(length_train_texts)) 
        print('Std dev is:', np.std(length_train_texts))
        print('mean+ sd.deviation value for train encoded text is:', '\n', int(mean(length_train_texts)) + int(np.std(length_train_texts)))
        self.max_length = int(mean(length_train_texts)) + int(np.std(length_train_texts))
        print("assigned max_length is:", self.max_length)
        #Padding the Integer Encoded Data to the max_length
        self.padded_train_data = pad_sequences(self.train_encoded_doc, maxlen=self.max_length) 
        print("Shape of Training Data is:", self.padded_train_data.shape, type(self.padded_train_data), len(self.padded_train_data),
        '\n', self.padded_train_data[0:5]) 
        print("Shape of Training Label is:", type(self.train_labels), len(self.train_labels))
    
    def bert(self):
        print('BERT START', str(datetime.datetime.now()))
        # A. Using Bert Model 
        bert_embedding = BertEmbedding(model='bert_24_1024_16', dataset_name='book_corpus_wiki_en_cased')
        self.result = bert_embedding(self.train_texts)
        print(type(self.result))
        print(self.result[0])
        id2emd = {}
        id2word = {}
        id_n = 1
        self.embedding_dim = 0
        sequences = []
        for (vocab_list, emb_list) in self.result:
            sequence = []
            for i in range(len(vocab_list)):
                if self.embedding_dim == 0:
                    self.embedding_dim = len(emb_list[i])
                sequence += [id_n]
                id2emd[id_n] = emb_list[i]
                id2word[id_n] = vocab_list[i]
                id_n += 1
            sequences += [sequence]
        # Creating embedding matrix
        keys = sorted(id2word.keys())
        self.embedding_matrix = np.zeros((id_n, self.embedding_dim))
        for id_key in keys:
            embedding_vector = id2emd[id_key]
            self.embedding_matrix[id_key] = embedding_vector
        print('# Embeddings loaded. Matrix size:', self.embedding_matrix.shape)
        print('MATRIX ELEMENTS', self.embedding_matrix[0:10])
        print('BERT LOADED', str(datetime.datetime.now()))
        self.vocab_size = id_n

    def word2vec(self):
        print('> loading word2vec embeddings')
        #B. Word2Vecvec Using Gensim
        word_vectors = KeyedVectors.load_word2vec_format('file_path/word2vec-GoogleNews-vectors-negative300.bin', binary=True)
        # Creating embedding matrix
        self.embedding_matrix = np.zeros((self.vocab_size, 300))
        for word, i in self.tokenizer.word_index.items():
            if word in word_vectors:
                embedding_vector = word_vectors[word]
                self.embedding_matrix[i] = embedding_vector
        del(word_vectors)
        print('MATRIX ELEMENTS', self.embedding_matrix[0:10])
       
    def glove(self):
        print('> loading glove embeddings')
        #C. Glove Using Gensim
        word_vectors = KeyedVectors.load_word2vec_format('file_path/glove.6B.300d.word2vec.txt', binary=False)
        # Creating embedding matrix
        self.embedding_matrix = np.zeros((self.vocab_size, 300))
        for word, i in self.tokenizer.word_index.items():
            if word in word_vectors:
                embedding_vector = word_vectors[word]
                self.embedding_matrix[i] = embedding_vector
                del(word_vectors)
        print('MATRIX ELEMENTS', self.embedding_matrix[0:10])

    def fasttext(self):
        print('> loading fasttext embeddings')
        #D. Fast Text Using Gensim 
        word_vectors = KeyedVectors.load_word2vec_format('file_path/fasttext-300d-2M.vec', binary=False)
        # Creating embedding matrix
        self.embedding_matrix = np.zeros((self.vocab_size, 300))
        for word, i in self.tokenizer.word_index.items():
            if word in word_vectors:
                embedding_vector = word_vectors[word]
                self.embedding_matrix[i] = embedding_vector
        del(word_vectors)
        print('MATRIX ELEMENTS', self.embedding_matrix[0:10])
        
    def domain_train(self):
        print('> loading domain embeddings')
        #E. Training the self word embedding
        word_vectors = KeyedVectors.load_word2vec_format('file_path/embedding_model.txt', binary=False)
        # Creating embedding matrix
        self.embedding_matrix = np.zeros((self.vocab_size, self.embedding_dim))
        for word, i in self.tokenizer.word_index.items():
            if word in word_vectors:
                embedding_vector = word_vectors[word]
                self.embedding_matrix[i] = embedding_vector
        del(word_vectors)
        print('MATRIX ELEMENTS', self.embedding_matrix[0:10])
        
    def lstm(self):
        self.model = Sequential()
        e = Embedding(self.vocab_size, self.embedding_dim, weights=[self.embedding_matrix], input_length=self.max_length, trainable=False)
        self.model.add(e)
        self.model.add(LSTM(128, return_sequences=True, dropout=0.2))
        self.model.add(LSTM(64, return_sequences=False, dropout=0.1)) 
        self.model.add(Dense(1, activation='sigmoid'))
        # Compiling the model
        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy']) 
        # Summarizing the model
        print(self.model.summary())

    def bi_lstm(self):
        self.model = Sequential()
        e = Embedding(self.vocab_size, self.embedding_dim, weights=[self.embedding_matrix], input_length=self.max_length, trainable=False) 
        self.model.add(e)
        self.model.add(Bidirectional(LSTM(128, return_sequences=True, dropout=0.1)))
        self.model.add(Bidirectional(LSTM(64, return_sequences=False, dropout=0.1)))
        self.model.add(Dense(16))
        self.model.add(Dense(1, activation='sigmoid'))
        # Compiling the model
        self.model.compile(optimizer='rmsprop', loss='binary_crossentropy', metrics=['accuracy'])  
        # Summarizing the model
        print(self.model.summary())

    # To reset the model method-2
    def reset_models(self, model_type,attention_layer):
        if model_type == 'lstm':
            self.lstm(attention_layer)
        elif model_type == 'lstm_cnn':
            self.lstm_cnn()
        elif model_type == 'bi_lstm':
            self.bi_lstm()

    def train(self,model_type,attention_layer):    
        X = self.padded_train_data  
        Y = np.array(self.train_labels) 
        # K-fold Validation 
        kf = KFold(n_splits=10, shuffle=False)
        kf.get_n_splits(X)
        acc = []
        p = []
        r = []
        f = []
        ba = []
        results = []
        x_train_text, x_test_text ,y_train_label,y_test_label = (None,None,None,None)
        for train_index, test_index in kf.split(X):
            print('',train_index[0:5], type(train_index))
            print(test_index[0:5], type(test_index))
            x_train_text, x_test_text=X[train_index], X[test_index] 
            y_train_label, y_test_label=Y[train_index], Y[test_index] 
            print('The shape of x_train_text and x_test_text are:', x_train_text.shape, x_test_text.shape)
            print('The type of x_train_text and x_test_text are:', type(x_train_text), type(x_test_text))
            print('The shape of y_train_label and y_test_label are:', y_train_label.shape, y_test_label.shape)
            print('The type of y_train_label and y_test_label are:', type(y_train_label), type(y_test_label))  
            gc.collect()
            self.model.fit(x_train_text, y_train_label, epochs=20, batch_size=64, verbose=1) 
            #Making predictions on test data
            print('Old evaluation:')
            pred_labels=self.model.predict_classes(x_test_text)
            print('-----The 1st Classification Report')
            print(classification_report(y_test_label, pred_labels, digits=4))
            print('-----The 1st Confusion Matrix')
            print('The confusion matrix is', '\n', confusion_matrix(y_test_label, pred_labels))
            print('\nOriginal classes:', y_test_label[:20], '\n', len(y_test_label), type(y_test_label))
            print('Predicted classes', pred_labels[:10], '\n', len(pred_labels), type(pred_labels))
            
            #Generating a CSV File of predicted results 
            pred=pd.DataFrame(columns=['ID', 'Orginal Labels', self.target_class])
            pred['ID'] = test_index
            pred['Orginal Labels'] = y_test_label
            pred[self.target_class] = pred_labels
            results += [pred]
            print('The data Frame pred results ', pred[:5])

            # Computing the first metrics :
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
            
            print('new_y_test_label:', new_y_test_label[:], '\n', type(new_y_test_label), len(new_y_test_label))
            print('new_pred_labels:', new_pred_labels[:], '\n', type(new_pred_labels), len(new_pred_labels))
            
            print('-----The 2nd Classification Report')
            print(classification_report(new_y_test_label, new_pred_labels, digits=4))

            print('-----The 2nd Confusion Matrix')
            print('The confusion matrix is', '\n', confusion_matrix(new_y_test_label, new_pred_labels))

            #Computing the new metrics :
            print('Computing the new metrics:')
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
            print('Caluclating the mean of the both metrics:')
            acc_binary = (acc_binary+new_acc_binary)/2
            p_binary = (p_binary+new_p_binary)/2
            r_binary = (r_binary+new_r_binary)/2
            f_binary = (f_binary+new_f_binary)/2
            b_acc = (b_acc+new_b_acc)/2
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
            # reset the models
            self.reset_models(model_type, attention_layer)

        #Printing Average Results    
        print('---- The final Averaged result after 10-fold validation: ' , self.target_class)
        print('>> Accuracy:', mean(acc)*100)
        print('>> Precision:', mean(p)*100)
        print('>> Recall:', mean(r)*100)
        print('>> F1:', mean(f)*100)
        print('>> Balanced Accuracy:', mean(ba)*100)
        pred_results = pd.concat(results, axis=0, join='inner').sort_index()   #Important Axis=0 means data will be joined coulmn to column, it mean for 10 fold there will be 10 coulmns while axis=0 is row addtion. so total rowx will  be 952 but columns will remain 1. 
        print(pred_results[0:20])
        pred_results.to_csv('/path' + self.target_class + '_pred_results.csv', index=False)

if __name__ == "__main__":
    print(sys.argv)
    parser =  argparse.ArgumentParser(description = "Arguments")
    parser.add_argument('--target-class', dest='target_class', default='Asthma', type=str, action='store', help='The bla bla')
    parser.add_argument('--word-embedding', dest='word_embedding', default='fasttext', type=str, action='store', help='The input file')
    parser.add_argument('--model-type', dest='model_type', default='bi_lstm', type=str, action='store', help='The input file')
    parser.add_argument('--attention-layer', dest='attention_layer', default='False', action='store', type=str, help='The input file')
    args = parser.parse_args()
    
#Step 1- Passing the target_class to the class with name of morbidity_obj
    morbidity_obj = morbidity(args.target_class)
    print(args.target_class)
    print(args.word_embedding)
    print(args.model_type)

#Step 2- Applying the method/function texts_and_labels 
    morbidity_obj.texts_and_labels()

#Step 3- Appyling the method/function padded_encoded_text 
    morbidity_obj.padded_encoded_text()
    
#Step 4- Applying the method/function to choose the type of word embedding 
    if args.word_embedding == 'word2vec': 
        morbidity_obj.word2vec()
    elif args.word_embedding == 'glove':
        morbidity_obj.glove()
    elif args.word_embedding == 'fasttext':
        morbidity_obj.fasttext()
    elif args.word_embedding == 'domain':
        morbidity_obj.domain_train()
    elif args.word_embedding == 'bert':
        morbidity_obj.bert()
    else:
        print('Please use one of them: BERT, Word2Vec, Glove, Fasttext or Domain')
        exit(1)
    #sys.exit(1)
  
#Step 5- Selecting a model to train        
    if args.model_type == 'lstm':
        morbidity_obj.lstm() 
    elif args.model_type == 'lstm_cnn':
        morbidity_obj.lstm_cnn()
    elif args.model_type == 'bi_lstm':
        morbidity_obj.bi_lstm()
    else:
        print('Please use one of models: lstm, lstm_cnn or bi_lstm')
        exit(1)
      
#Step 6- Applying the method/function train 
    morbidity_obj.train(args.model_type, args.attention_layer)
