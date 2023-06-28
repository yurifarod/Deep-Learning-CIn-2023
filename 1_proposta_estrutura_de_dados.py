#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 23:26:55 2023

@author: yurifarod
"""

import re
import csv
import nltk
import string
import numpy as np
import pandas as pd


'''
Proposta para transformar o texto em uma matriz de dados numericos segundo as palavras nele contidas e suas posicoes!

Este podera ser o insumo de treinamento da nossa rede neural. Isso nos permitira fugir de uma abordagem recorrente (mais complexa)

Nesta primeira parte definirei as funcoes necessarias!
'''

nltk.download('rslp')
stemmer = nltk.stem.RSLPStemmer()

def fu_encode(term, bag_of_words):
    try:
        index = bag_of_words.index(term)
    except:
        bag_of_words.append(term)
        index = len(bag_of_words) - 1
    
    return index

def fu_decode(index, bag_of_words):
    term = bag_of_words[index]
    
    return term

def codificar(text, bag_of_words):
    zero_list = text.split()
    
    first_list = []
    for i in zero_list:
        if len(i) > 2 and (not i.isdigit()):
            word = stemmer.stem(i)
            first_list.append(fu_encode(word, bag_of_words))
    
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

df = pd.read_csv('dataset/novas_categorias.csv')

'''
Existem alguns poucos textos nulos, vamos remover eles!

'''
df_dataset = df.dropna(subset=['text'])

'''
Balanceamento da base de dados de treino com a quantidade de notícias da menor classe
'''


#Bases montadas e retipadas para execução do restante do código
df_dataset = df_dataset.sort_values('category')

df_balanced = df_dataset.loc[df['category'] == 'meio-ambiente, saude e bem-estar']

max_size = len(df_balanced)

categorias = ['cultura e turismo', 'educacao, ciencia e tecnologia', 'internacional', 'politica', 'cotidiano', 'esporte']

for cat in categorias:
    df_temp = df_dataset.loc[df['category'] == cat]
    df_temp = df_temp.sample(n = max_size)
    df_balanced = pd.concat([df_balanced, df_temp])

df_dataset = df_balanced
'''
Finalizada a montagem da estrutura que será utilizada
'''
df_dataset = df_dataset.drop(columns='category')


df_dataset = df_dataset.values.tolist()


'''
Nosso texto estara no registro [1] e as classes nos [2..7]. Vamos remover primeiros as pontuacoes e stopwords
'''

nltk.download('stopwords')
stopwords = nltk.corpus.stopwords.words('portuguese')
punctuation = string.punctuation
padrao_sw = re.compile(r'\b(' + r'|'.join(stopwords) + r')\b\s*')

'''
Montagem da Bag of Words que auxiliara no hash para os termos
'''
bag_of_words = []

#Treino
nr_reg = 0
for i in df_dataset:
    text = i[1]
    text = padrao_sw.sub('', text)
    text = re.sub(r'[^\w\s]', '', text)
    text = text.upper()
    encoded = codificar(text, bag_of_words)
            
    file = 'news_nr_'+f'{nr_reg:08}'+'.csv'
    filename = './dataset_matrix/' + file
    
    encoded = np.asarray(encoded)
    pd.DataFrame(encoded).to_csv(filename)
    
    class_file = './dataset_class/class_file.csv'
    with open(class_file, 'a') as csv_file: 
        write = csv.writer(csv_file)
        write.writerow([file, i[2], i[3], i[4], i[5], i[6], i[7], i[8]])
    
    nr_reg += 1
    
'''
A estrutura de dados Matricial foi gerada, o primeiro passo antes de aplicá-la a um classificador é aplicar o MinMaxScaler
Pode seguir o modelo em:
https://github.com/yurifarod/Algorithms/blob/master/Curso%20Udemy%20RNA/Previs%C3%A3o%20da%20Bolsa%20(Recorrente)/petr4-recorrente-multiplas-saidas.py
Nas bases de treino e teste (antes de tudo)
'''