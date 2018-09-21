# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 08:18:36 2018

@author: huynheri
"""
import os
import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import eli5
from eli5.sklearn import PermutationImportance

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

#change directory to current folder
os.chdir('C:\\Users\\' + os.getlogin() + '\\Documents\\GitHub\\Kaggle Learnings')

#%%
#basic XGBoost classifier application on health dataset
#XGBoost Notes
#start with high n values (number of times to go through the cycle) and set a early_stopping_rounds
#so the model knows to stop when the validation score doesn't improve anymore (early_stopping_round
#in the fit method)

"""Pregnancies#Number of times pregnant
GlucosePlasma# glucose concentration a 2 hours in an oral glucose tolerance test
BloodPressureDiastolic# blood pressure (mm Hg)
SkinThicknessTriceps# skin fold thickness (mm)
Insulin2-Hour# serum insulin (mu U/ml)
BMIBody# mass index (weight in kg/(height in m)^2)
DiabetesPedigreeFunction#Diabetes pedigree function
Age#Age (years)
Outcome#Class variable (0 or 1)"""

column = np.array(['Pregnancies','GlucosePlasma','BloodPressure','SkinThickness','Insulin','BMIBody',
          'DiabetesPedigreeFunction','Age','Outcome'])
data = pd.read_csv('input/pima-indians-diabetes.data.csv',names=column)
data.head()

X = data.loc[:,column[column!='Outcome']]
y = data.loc[:,'Outcome']

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=seed)

X_matrix = X.values
y_matrix = y.values
X_test_matrix = X_test.values
y_test_matrix = y_test.values
xgb = XGBClassifier()
xgb.fit(X_matrix, y_matrix,verbose=False)

#xgb = XGBClassifier(learning_rate=0.05,n_estimators=1000,n_jobs=4)
#xgb.fit(X_matrix, y_matrix,early_stopping_rounds=5,eval_set=[(X_test_matrix, y_test_matrix)], verbose=False)
accuracy_score(y_test_matrix,xgb.predict(X_test_matrix))

#%%basic implementation of permutation importance
#notes from permutation importance
#if two features share the same weight, they are most likely entangled and feature engineering
#should be performed

perm_xgb = PermutationImportance(xgb).fit(X_matrix, y_matrix)
permutation_html = eli5.show_weights(perm_xgb, feature_names=list(X))

html = permutation_html.data
with open('permutation.html', 'w') as f:
    f.write(html)
    



