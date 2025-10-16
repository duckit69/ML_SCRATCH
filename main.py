import numpy as np 
import pandas as pd


# 1- Load Data
x_train = pd.read_csv("train.csv")

print(x_train.isnull().sum())
# x_train.isnull().sum() Returns a Series showing the count of missing values in each column.
# x_train.shape works like np.shape returns rows, cols
# Age with 177 missing values + Cabin 687 + embarked 2
# i will use Age + sex

# 2- Clean/prepocess Data

# for Age i will fill missing values with median 

x_train = x_train[['Age', 'Sex']]
print(x_train.head())
