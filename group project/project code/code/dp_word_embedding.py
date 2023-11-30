import time

from docutils.nodes import attention
from gensim.models.keyedvectors import KeyedVectors

#Pandas and Numpy
import pandas as pd
import numpy as np
from statistics import mean

#Keras
from keras.models import Sequential
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense,Embedding, LSTM, Bidirectional

#Sci-Kit Library
from sklearn.metrics import  roc_curve, auc
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score

#Miscellaneous
import argparse

class morbidity                                                                                                                                                                     :
    def __init__(self, target_class):
        self.target_class = target_class
        self.train_data = pd.read_csv('../data/processed/' + target_class + '/'+ target_class + '.csv')
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
        self.embedding_dim = 300

    def texts_and_labels(self):
        df = self.train_data[['id', 'text', 'label']]

        df0 = df.loc[df['label'] == 0]
        n0 = len(df0)

        df1 = df.loc[df['label'] == 1]
        n1 = len(df1)

        print('Labels distribution of Training Labels:', '\n', 'Zeros =', n0, 'Ones =', n1)

        if (n0 > n1):
            df1 = pd.DataFrame(np.repeat(df1.values, n0//n1, axis=0), columns=['id', 'text', 'label'])
        elif (n1 > n0):
            df0 = pd.DataFrame(np.repeat(df0.values, n1 // n0, axis=0), columns=['id', 'text', 'label'])

        self.train_data = pd.concat([df0, df1], ignore_index=True)

        texts = []
        labels = []
        for i, r in self.train_data.iterrows():
            texts += [r['text'].strip().split('\n', 1)[1]]
            labels += [r['label']]
        self.train_texts = texts
        self.train_labels = labels

    def padded_encoded_text(self):
        # Tokenizing the Data 
        self.tokenizer.fit_on_texts(self.train_texts)
        # Defining the length of vocabulary
        self.vocab_size = len(self.tokenizer.word_index) + 1
        # Defining the vocabulary made from unique words
        self.my_vocab = set([w for (w,i) in self.tokenizer.word_index.items()])
        #Encoding the data to integar
        self.train_encoded_doc = self.tokenizer.texts_to_sequences(self.train_texts)
        # Calculating the average, standard deviation & maximum length of Encoded Training Data
        length_train_texts = [len(x) for x in self.train_encoded_doc]
        self.max_length = int(mean(length_train_texts)) + int(np.std(length_train_texts))
        #Padding the Integer Encoded Data to the max_length
        self.padded_train_data = pad_sequences(self.train_encoded_doc, maxlen=self.max_length)

    def word2vec(self):
        print('> loading word2vec embeddings')
        #B. Word2Vecvec Using Gensim
        word_vectors = KeyedVectors.load_word2vec_format('../lib/word2vec-GoogleNews-vectors-negative300.bin', binary=True)
        # Creating embedding matrix
        self.embedding_matrix = np.zeros((self.vocab_size, 300))
        for word, i in self.tokenizer.word_index.items():
            if word in word_vectors:
                embedding_vector = word_vectors[word]
                self.embedding_matrix[i] = embedding_vector
        del(word_vectors)
       
    def glove(self):
        print('> loading glove embeddings')
        #C. Glove Using Gensim
        word_vectors = KeyedVectors.load_word2vec_format('../lib/glove.6B.300d.w2vformat.txt', binary=False)
        # Creating embedding matrix
        self.embedding_matrix = np.zeros((self.vocab_size, 300))
        for word, i in self.tokenizer.word_index.items():
            if word in word_vectors:
                embedding_vector = word_vectors[word]
                self.embedding_matrix[i] = embedding_vector
                del(word_vectors)

    def fasttext(self):
        print('> loading fasttext embeddings')
        #D. Fast Text Using Gensim 
        word_vectors = KeyedVectors.load_word2vec_format('../lib/fasttext-300d-2M.vec', binary=False)
        # Creating embedding matrix
        self.embedding_matrix = np.zeros((self.vocab_size, 300))
        for word, i in self.tokenizer.word_index.items():
            if word in word_vectors:
                embedding_vector = word_vectors[word]
                self.embedding_matrix[i] = embedding_vector
        del(word_vectors)

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

    def classification_metrics(self, Y_pred, Y_true):
        acc = accuracy_score(Y_true, Y_pred)
        fpr, tpr, thresholds = roc_curve(Y_true, Y_pred)
        auc_ = auc(fpr, tpr)
        precision = precision_score(Y_true, Y_pred)
        recall = recall_score(Y_true, Y_pred)
        f1 = f1_score(Y_true, Y_pred, zero_division=1)
        return acc, auc_, precision, recall, f1

    def swap_label(self, Y_test, Y_pred):
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

    def train(self):
        X = self.padded_train_data  
        Y = np.array(self.train_labels) 
        # K-fold Validation 
        kf = KFold(n_splits=5, shuffle=True)
        kf.get_n_splits(X)
        acc_t = []
        auc_t = []
        f1_t = []

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            Y_train, Y_test = Y[train_index], Y[test_index]
            self.model.fit(X_train, Y_train, epochs=1, batch_size=64, verbose=1)
            Y_pred=(self.model.predict(X_test) > 0.5).astype("int32")
            new_y_test, new_y_pred = self.swap_label(Y_test, Y_pred)

            old_acc, old_auc, old_precision, old_recall, old_f1 = self.classification_metrics(Y_pred, Y_test)
            new_acc, new_auc, new_precision, new_recall, new_f1 = self.classification_metrics(new_y_pred, new_y_test)

            acc_t.append(mean([old_acc, new_acc]))
            auc_t.append(mean([old_auc, new_auc]))
            f1_t.append(mean([old_f1, new_f1]))

            self.bi_lstm()

        acc_avg = mean(acc_t)
        auc_avg = mean(auc_t)
        f1_avg = mean(f1_t)
        return acc_avg, auc_avg, f1_avg

if __name__ == "__main__":
    method = input("Word embedding method:")
    diseases = ['Asthma', 'CAD', 'CHF', 'Depression', 'Diabetes', 'Gallstones', 'GERD', 'Gout', 'HC', 'Hypertension',
                'HT', 'OA', 'Obesity', 'OSA', 'PVD', 'VI']
    if method == "":
        method = "word2vec"
    print("---------------------------")
    print('Word embedding method:' + method)
    avg = []
    for disease in diseases:
        print("---------------------------")
        print('Disease Name:' + disease)
        parser = argparse.ArgumentParser(description="Arguments")
        parser.add_argument('--target-class', dest='target_class', default=disease, type=str, action='store',
                            help='The bla bla')
        parser.add_argument('--word-embedding', dest='word_embedding', default=method, type=str, action='store',
                            help='The input file')
        parser.add_argument('--model-type', dest='model_type', default='bi_lstm', type=str, action='store',
                            help='The input file')
        parser.add_argument('--attention-layer', dest='attention_layer', default='False', action='store', type=str,
                            help='The input file')
        args = parser.parse_args()

        morbidity_obj = morbidity(args.target_class)

        morbidity_obj.texts_and_labels()

        morbidity_obj.padded_encoded_text()

        if args.word_embedding == 'word2vec':
            morbidity_obj.word2vec()
        elif args.word_embedding == 'glove':
            morbidity_obj.glove()
        elif args.word_embedding == 'fasttext':
            morbidity_obj.fasttext()
        else:
            print('Please use one of them: Word2Vec, Glove, Fasttext')

        if args.model_type == 'bi_lstm':
            morbidity_obj.bi_lstm()
        else:
            print('Please use one of models: lstm, lstm_cnn or bi_lstm')

        start_time = time.time()
        acc_avg, auc_avg, f1_avg = morbidity_obj.train()
        avg.append([acc_avg, auc_avg, f1_avg])
        print('-----The Metrics Report------')
        print('>>> Training Time:', time.time() - start_time)
        print('>>> Accuracy:', acc_avg)
        print('>>> AUC:', auc_avg)
        print('>>> F1:', f1_avg)

    print('-----The Final Report------')
    average = np.array(avg)
    column_mean = np.mean(average, axis=0)
    print('>>> Average Accuracy:', column_mean[0])
    print('>>> Average AUC:', column_mean[1])
    print('>>> Average F1:', column_mean[2])