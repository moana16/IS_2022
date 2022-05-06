# -*- coding: utf-8 -*-
"""
Created on Fri May  6 15:36:23 2022

@author: elizd
"""
import pandas as pd

X = pd.DataFrame()

X['gender'] = ['F','M','F','F','M']
X
X['age']=[15, 35, 25, 37, 55]

from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler

encoder = OneHotEncoder(sparse=False,
                        handle_unknown='ignore')
scaler = MinMaxScaler()

from sklearn.compose import ColumnTransformer

obj_columns = ['gender']
num_columns=['age']
ct = ColumnTransformer(
    [('scaler', scaler, num_columns),
     ('encoder', encoder, obj_columns)])
ct.fit(X)

print(X)
print(ct.transform(X))


















