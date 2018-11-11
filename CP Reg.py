# -*- coding: utf-8 -*-
"""
Created on Sat Oct 13 16:43:37 2018

@author: quinn
"""

#importing libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
 

#importing dataset
dataset = pd.read_csv('CP Mult Reg Data.csv')
X = dataset.iloc[:, :-1].values
Y = dataset.iloc[:, 10].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X1 = LabelEncoder() 
X[:, 1] = labelencoder_X1.fit_transform(X[:, 1])
onehotencoder = OneHotEncoder(categorical_features =[1])
X = onehotencoder.fit_transform(X).toarray()

labelencoder_X5 = LabelEncoder() 
X[:, 5] = labelencoder_X5.fit_transform(X[:, 5])

labelencoder_X6 = LabelEncoder() 
X[:, 6] = labelencoder_X6.fit_transform(X[:, 6])
 
labelencoder_X7 = LabelEncoder() 
X[:, 7] = labelencoder_X7.fit_transform(X[:, 7])
  
labelencoder_X8 = LabelEncoder() 
X[:, 8] = labelencoder_X8.fit_transform(X[:, 8])

   
#Splitting into Training and Test

from sklearn.model_selection import train_test_split
X_train, X_test, Y_Train, Y_Test = train_test_split(X, Y, test_size =0.2)

# Fitting MLR to TrainingSet    

from sklearn.linear_model import LinearRegression
lm=LinearRegression()
lm.fit(X_train, Y_Train)

#Jose

print(lm.intercept_)
lm.coef_


# Predicting the Test set results
y_pred = lm.predict(X_test)



#adding column of ones
import statsmodels.formula.api as sm
X=np.append(arr = np.ones((72, 1)).astype(int), values = X, axis = 1)

# Backward Elimination

X_opt = X[:, [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]]
lm_OLS = sm.OLS(endog = Y, exog = X_opt).fit()
lm_OLS.summary()
