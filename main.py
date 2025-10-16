import numpy as np 
import pandas as pd


x_train = pd.read_csv("train.csv")

print(x_train.info())

#
x_train['Age'] = x_train['Age'].fillna(x_train['Age'].mean()) 
print(x_train['Age'].value_counts().sort_index())

# number of survivors for each cabin
survival_by_cabin = x_train.groupby('Cabin')['Survived'].sum()

#print(survival_by_cabin)