#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 26 17:16:14 2023

@author: yurifarod
"""

import pandas as pd

df = pd.read_csv('dataset/articles.csv')

category = df.groupby(['category'])['title'].count()

category.to_csv('dataset/categorias.csv')