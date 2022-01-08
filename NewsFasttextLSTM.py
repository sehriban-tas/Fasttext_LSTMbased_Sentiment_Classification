# -*- coding: utf-8 -*-

import re
import os
import numpy as np
import yaml
import xlrd as excel
import numpy as n
import pandas as pd
import sklearn
from twitter_preprocessor import TwitterPreprocessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from keras.utils import to_categorical
import keras.backend as K
from keras.layers import *
from keras.losses import *
from keras.models import *
from keras.callbacks import *
from keras.activations import *
from keras.models import Sequential
from keras.layers import Embedding, InputLayer, SimpleRNN, LSTM,GRU, Bidirectional,TimeDistributed,Dense,Input,Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing import sequence
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import WordPunctTokenizer
import matplotlib.pyplot as plt
import seaborn as sns


"""-------------------------------------------------------"""
def plotResults(history):
    acc = history.history['sparse_categorical_accuracy']
    val_acc = history.history['val_sparse_categorical_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(len(acc))

    plt.plot(epochs, acc, 'r', label='Acc')
    plt.plot(epochs, val_acc, 'g', label='Val_Acc')
    plt.title('Eğitim ve Doğrulama Doğruluk Değeri')
    plt.legend()

    plt.figure()

    plt.plot(epochs, loss, 'r', label='Loss')
    plt.plot(epochs, val_loss, 'g', label='Val_Loss')
    plt.title('Eğitim ve Doğrulama Hatası')
    plt.legend()

    plt.show()
"""-------------------------------------------------------"""
"""-------------------------------------------------------"""

def prepare_tokenizer_and_weights(X):
    tokenizer = Tokenizer(filters='file_name')
    tokenizer.fit_on_texts(X)
    
    weights = np.zeros((len(tokenizer.word_index)+1, 300))
    with open("H:\\Staj\\FastTextEmbeddingsTr\\cc.tr.300.vec",encoding="utf8") as f:
        next(f)
        for l in f:
            w = l.split(' ')
            if w[0] in tokenizer.word_index:
                weights[tokenizer.word_index[w[0]]] = np.array([float(x) for x in w[1:301]])
                # print("tokk",tokenizer)
                
    return tokenizer, weights
"""-------------------------------------------------------"""
def readRawData(folder):
    #returns the names of the files in the directory data as a list
    path="H:\\Staj\\TTC-3600_Orj\\"
    path=path+folder+"\\"
    
    list_of_files = os.listdir(path)
    # print("/n***",list_of_files)
    docs=[]
    for file in list_of_files:
        print(path+file)
        f = open(path+file, 'r', encoding='utf-8')
        #append each line in the file to a list
        lines=f.readlines()
        content=""
        for line in lines:
            content=content+line.strip()     
        # print(content)
        docs.append(content)
       
        f.close()    
        
    return docs
"""-------------------------------------------------------"""
folder_name=["ekonomi","kultursanat","saglik","siyaset","spor","teknoloji"]
#folder_name=["saglik"]

all_docs=[]
labels=[]
i=0
for folder in folder_name:
    docs=readRawData(folder)
    i=i+1
    for item in docs:
        labels.append(i)
        #print("Öncesi:")
        #print(item)
        item=item.replace("İ", "i")
        item=item.replace("", "'")
        item=item.replace("", "")
        p = TwitterPreprocessor(item)
        #tweets.append(p.fully_preprocess().text)
        content=p.fully_preprocess().text
        content = re.sub('[‘“’”\"\'´]', '', content)
        #print ("Sonrası:")
        #print(content)
        all_docs.append(content.strip())

print('File reading successful...')

#------TRAİN DATA------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(all_docs, labels, test_size=0.2)
# print("test",X_test)


all_docs_token, all_weights = prepare_tokenizer_and_weights(X_train+X_test)

# print("all_we",all_weights)
# print("all_token",all_docs_token)//all_token <keras_preprocessing.text.Tokenizer object at 0x000001958504A748>
all_docs_sequences = all_docs_token.texts_to_sequences(X_train+X_test)
# print("seqqqqqq***",all_docs_sequences)
MAX_LEN = max(map(lambda x: len(x), all_docs_sequences))
print('all_docs_sequences is ok')
print('Max Len: ',MAX_LEN)
max_words = 10000  # We will only consider the top 10,000 words in the dataset

train_docs_sequences = all_docs_token.texts_to_sequences(X_train)
# print("trainseq",train_docs_sequences)

MAX_ID = len(all_docs_token.word_index)  

word_index = all_docs_token.word_index
print('Train Found %s unique tokens.' % len(word_index))
data = pad_sequences(train_docs_sequences, maxlen=MAX_LEN)
print("slkdf",data)
X_seq_train = data
labels = np.asarray(y_train)

print('Train Shape of train data tensor:', data.shape)
print('Train Shape of train label tensor:', labels.shape)

y_train = np.asarray(y_train)
#------TEST DATA------------------------------------------

test_docs_token, test_weights = prepare_tokenizer_and_weights(X_test)
test_docs_sequences = all_docs_token.texts_to_sequences(X_test)
# print ("lksdjflsdf",test_docs_sequences)

word_index = test_docs_token.word_index
print('Test Found %s unique tokens.' % len(word_index))
data = pad_sequences(test_docs_sequences, maxlen=MAX_LEN)
X_seq_valid = data
labels = np.asarray(y_test)

print('Test Shape of train data tensor:', data.shape)
print('Test Shape of train label tensor:', labels.shape)
y_valid = np.asarray(y_test)

def make_fast_text():#MODEL OLUŞTURMA
    fast_text = Sequential()
    # fast_text.add(InputLayer((MAX_LEN))) 
    fast_text.add(Embedding(input_dim=MAX_ID+1, output_dim=300, weights=[all_weights], trainable=True))
    fast_text.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    fast_text.add(Dense(6,activation='softmax'))
    return fast_text

fast_texts = [make_fast_text() for i in range(1)]

fast_texts[0].summary()
# print("kjdflksjdflsdf",fast_texts)//[<keras.engine.sequential.Sequential object at 0x0000016697B5EFC8>]

for fast_text in fast_texts:
    #X_seq_train, X_seq_valid, y_train, y_valid = train_test_split(data, labels2, test_size=0.2)
    #MODELİN DERLENMESİ,karşılaştırma yani loss; 2den fazla class ve int encodin old SC crosentropy loss olarak seçilir. metrics=etiket ve tahmin uyuşma oranın gösterir.
    #
    fast_text.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['sparse_categorical_accuracy'])
    #MODELİN EĞİTİLMESİ 
    history=fast_text.fit(X_seq_train, y_train, validation_data=(X_seq_valid, y_valid),
                 callbacks=[EarlyStopping(monitor='val_loss', patience=15, verbose=0)],
                 batch_size=128,
                 epochs=3)
    
    plotResults(history)
