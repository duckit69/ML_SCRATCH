import numpy as np 
import pandas as pd

# 1- Load Data
x_train = pd.read_csv("train.csv")
# print(x_train.info())
# x_train.shape works like np.shape returns rows, cols
# Age with 177 missing values + Cabin 687 + embarked 2
# i will use Age + sex

# 2- Clean/prepocess Data

x_train = x_train[['Age', 'Sex']]
# find missing values
# isnull() put as True every value not from the same Type, sum all Trues
# print(x_train.isnull().sum())

# for Age i will fill missing values with median 
x_train['Age'] = x_train['Age'].fillna(x_train['Age'].median())
# for sex use 1/0
x_train['Sex'] = x_train['Sex'].map({'male' : 1, 'female' : 0})
