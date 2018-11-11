# -*- coding: utf-8 -*-
"""
Created on Tue Aug 21 16:44:45 2018

@author: quinn
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#importing dataset
dataset = pd.read_csv('Copy of Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 7].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder() 
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features =[1])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_X5 = LabelEncoder() 
X[:, 5] = labelencoder_X5.fit_transform(X[:, 5])

labelencoder_X6 = LabelEncoder() 
X[:, 6] = labelencoder_X6.fit_transform(X[:, 6])
 
labelencoder_Y = LabelEncoder()
Y=labelencoder_Y.fit_transform(Y)    

#Splitting into Training and Test

from sklearn.model_selection import train_test_split
X_train, X_test, Y_Train, Y_Test = train_test_split(X, Y, test_size =0.2)

#Fitting Logistic to TrainingSet
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression()
classifier.fit(X_train, Y_Train)

#Predicting the Test Set Results
y_pred = classifier.predict(X_test)

#Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(Y_Test, y_pred)

cm
print(classifier.intercept_)
classifier.coef_
from sklearn import metrics
 
 