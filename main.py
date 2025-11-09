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


# 3- Standardize features
# i want to make the age between 0-1 minMax (value - min) / (max - min)
#, Sex is already 0/1

def minMax(x, min, max):
    return ((x - min) / (max - min))
min_x = x_train['Age'].min()
max_x = x_train['Age'].max()
x_train['Age'] = x_train['Age'].apply(minMax, args=(min_x, max_x,)) 


# 4- tools used for logistic regression model

#sigmoid 
def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# cost function
def cost_funct(m, y_train, x_train, w, b):
    z = x_train.dot(w) + b
    left_part = y_train * np.log(sigmoid(z))
    right_part = (1 - y_train) * np.log(1 - sigmoid(z))
    full_part = -1 * np.sum(left_part + right_part) / m
    return full_part

