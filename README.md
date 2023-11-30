# CSE6250 Big Data Analytics for Healthcare 

### Team ID: D4

### Team members: Yuyu Yao, Chengqi Huang

```
This repository contains the source code used for our group project. 
```

### System Requirements:

```
Python Packages:
python3
numpy
pandas
scikit-learn
gensim
nltk

Word Embedding Packages:
pre-trained Word2Vec8 model
pre-trained GloVe6B10 embeddings model 
Facebook proposed fastText
```

### Access to the dataset can be requested by the website of Harvard Medical School:

```
URL https://portal.dbmi.hms.harvard.edu/projects/n2c2-nlp/
```

### Download Instructions:

~~~~~~
Source and preprocessed data:
|--data
  |--fsttext
  |--glove
  |--word2vec
  |--source
  |--processed
~~~~~~

~~~~~~
Code for training CML model:
|--code
  |--CML_bag_of_word.py
  |--CML_word_embedding.py 
  
Data folder is required for implementing CML training using following two code files.

CML_bag_of_word.py: train CML model using processed csv data for each disease using TF-IDF method
--input: 
feature selection type should be one of "All Features"(default if not specified)/"ExtraTreesClassifier"/"SelectKBest"
--output: 
Printed classification metrics (ACC, AUC, F1) for 16 diseases.

CML_word_embedding.py: train CML model using processed csv data for each disease using word-embedding method
--input: 
Word embedding method should be one of "word2vec"(default if not specified)/"fasttext"/"glove"
--output: 
Printed classification metrics (ACC, AUC, F1) for each 16 diseases.
~~~~~~

~~~~~~
Code for training DL model:
|--code
  |--DL_word_embedding.py 
  
Data folder is required for implementing DL model training using following two code files.

DL_word_embedding.py: train DL model using processed csv data for each disease using word-embedding method
--input: 
Word embedding method should be one of "word2vec"(default if not specified)/"fasttext"/"glove"
--output: 
Printed classification metrics (ACC, AUC, F1) for each 16 diseases.
~~~~~~

~~~~~~
Code for preprocessing data:
|--code
  |--xml2csv_x.py
  |--xml2csv_y.py
  |--word_embedding.py

xml2csv_x.py: convert xml source data to csv data, and extract medical record with labels
xml2csv_y.py: convert xml source data to csv data, and extract meaningful labels
word_embedding.py: extract feature and label using word embeddding methods. 
word embedding packages are required to run this python file
--input: 
Word embedding method should be one of "word2vec"(default if not specified)/"fasttext"/"glove".
--output:
data.csv for each disease in each word embedding folder
~~~~~~

