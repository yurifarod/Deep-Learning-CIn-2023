#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May  3 23:06:27 2023

@author: yurifarod
"""

import pandas as pd
from sklearn.preprocessing import OneHotEncoder

df = pd.read_csv('dataset/articles.csv')

df['category'].to_frame().value_counts().plot(kind='bar')

df['category'] = df['category'].replace(['2015', '2016', 'banco-de-dados', 'cenarios-2017', 'colunas',
                                         'cotidiano', 'especial', 'folhinha', 'sobretudo', 'topofmind',
                                         'asmais', 'infograficos', 'paineldoleitor', 'multimidia'], 'cotidiano')

df['category'] = df['category'].replace(['ambiente', 'bichos', 'comida', 'equilibrioesaude', 'mulher',
                                         'treinamento', 'treinamentocienciaesaude'], 'meio-ambiente, saude e bem-estar')

df['category'] = df['category'].replace(['ilustrada', 'ilustrissima', 'musica', 'o-melhor-de-sao-paulo',
                                         'saopaulo', 'serafina', 'turismo', 'tv', 'vice'], 'cultura e turismo')

df['category'] = df['category'].replace(['ciencia', 'contas-de-casa', 'educacao', 'empreendedorsocial',
                                         'guia-de-livros-discos-filmes', 'guia-de-livros-filmes-discos',
                                         'seminariosfolha', 'tec'], 'educacao, ciencia e tecnologia')

df['category'] = df['category'].replace(['bbc', 'dw', 'euronews', 'mundo', 'rfi', 'mundo'], 'internacional')


df['category'] = df['category'].replace(['mercado', 'opiniao', 'poder', 'ombudsman'], 'politica')


#df['category'].to_frame().value_counts().plot(kind='bar')

df = df.drop(columns='title')

df = df.drop(columns='date')

df = df.drop(columns='subcategory')

df = df.drop(columns='link')

# creating instance of one-hot-encoder
enc = OneHotEncoder(handle_unknown='ignore')

enc_df = pd.DataFrame(enc.fit_transform(df[['category']]).toarray())
df = df.join(enc_df)

df.to_csv('dataset/novas_categorias.csv')