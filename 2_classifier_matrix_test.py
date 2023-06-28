#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 19:29:35 2023

@author: yurifarod
"""

import os
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

'''
Montando o dataset de Treino
'''
folder = './dataset_matrix/'

entries = os.listdir(folder)

entries.sort()

train_x = []
for news in entries:
    news_arq = folder + news
    df_train = pd.read_csv(news_arq, header=None, index_col=False, sep=',')
    df_train = df_train.iloc[1:]
    df_train = df_train.drop(columns=[0])
    serie = df_train.values.tolist()
    train_x.append(serie)


class_file = './dataset_class/class_file.csv'
df = pd.read_csv(class_file, header=None, index_col=False, sep=',')

df = df.drop(columns=[0])

y_train_orig = df.values.tolist()

train_x = np.array(train_x)


'''
Fazendo normalização dos dados com a formula de min-max
'''
max_value_word = 0
max_value_oc = 0
for reg in train_x:
    for line in reg:
        if line[0] > max_value_word:
            max_value_word = line[0]
        if line[1] > max_value_oc:
            max_value_oc = line[1]
            
for reg in train_x:
    for line in reg:
        line[0] = line[0]/max_value_word
        line[1] = line[1]/max_value_oc

'''
Agora vamos iniciar o treinamento efetivo da rede neural profunda!
'''
x_train = []
for reg in train_x:
    x_train.append(reg.reshape(-1).tolist())

#Definimos um metodo de criacao da RNA para ser usado posteriormente
classificador = Sequential()
classificador.add(Dense(units = 6000, activation = 'relu'))
classificador.add(Dropout(0.25))
classificador.add(Dense(units = 1024, activation = 'relu'))
classificador.add(Dropout(0.25))
classificador.add(Dense(units = 512, activation = 'relu'))
classificador.add(Dropout(0.25))
classificador.add(Dense(units = 256, activation = 'relu'))
classificador.add(Dense(units = 7, activation = 'sigmoid'))
classificador.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

acc = []
for i in range(10):
    X_train, X_test, y_train, y_test = train_test_split(x_train, y_train_orig, test_size=0.10)
    classificador.fit(X_train, y_train, batch_size = 10, epochs = 35)
    
    y_pred_base = classificador.predict(X_test)
    y_pred_max = np.argmax(y_pred_base, axis=1)
    y_test = np.argmax(y_test, axis=1)
    acc.append(accuracy_score(y_test, y_pred_max))

def Average(lst):
    return sum(lst) / len(lst)

print('Max Accuracy Obtained: '+str(max(acc)))
print('Min Accuracy Obtained: '+str(min(acc)))
print('Mean Accuracy Obtained: '+str(Average(acc)))