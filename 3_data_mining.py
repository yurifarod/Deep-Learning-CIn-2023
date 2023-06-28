#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jun 28 09:16:09 2023

@author: yurifarod
"""

import re
import nltk
import string
import numpy as np
import pandas as pd

def mining(text, bag_of_words):
    zero_list = text.split()
    
    first_list = []
    for i in zero_list:
        if len(i) > 2 and (not i.isdigit()):
            first_list.append(fu_mining(i, bag_of_words))
    
    second_list = []
    for term in first_list:
        if term not in second_list:
            second_list.append(term)
    
    first_matrix = np.zeros((3000,2))
    k = 0
    for term in second_list:
        first_matrix[k][0] = term
        
        for i in range(len(first_list)):
            if term == first_list[i]:
                first_matrix[k][1] += 1
        k += 1
        
    return first_matrix

def fu_mining(term, bag_of_words):
    try:
        index = bag_of_words.index(term)
    except:
        bag_of_words.append(term)
        index = len(bag_of_words) - 1
    
    return index

#Importacao de recursos aplicaveis ao processamento de linguagem natural
nltk.download('rslp')
nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')
punctuation = string.punctuation
padrao_sw = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')

df = pd.read_csv('dataset/novas_categorias.csv', nrows=1061)

df_dataset = df.drop(columns='category')

df_dataset = df_dataset.dropna(subset=['text'])

df_dataset = df_dataset.values.tolist()

'''
Montagem da Bag of Words que auxiliara no hash para os termos
'''

#Treino
id_news = 0
for i in df_dataset:
    bag_of_words = []
    text = i[1]
    text = padrao_sw.sub('', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.upper()
    encoded = mining(text, bag_of_words)
    
    encoded = encoded[encoded[:,1].argsort()[::-1]]
    
    print("#### Noticia "+str(id_news)+" ####")
    for j in range(5):
        index = int(encoded[j][0])
        print(bag_of_words[index])
    
    id_news += 1
        