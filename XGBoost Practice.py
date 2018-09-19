# -*- coding: utf-8 -*-
"""
Created on Wed Sep 19 08:18:36 2018

@author: huynheri
"""

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import eli5
from eli5.sklearn import PermutationImportance

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

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

model = XGBClassifier(silent=False)
my_model = model.fit(X_train, y_train)

accuracy_score(y_test,model.predict(X_test))

"""perm = PermutationImportance(my_model, random_state=1).fit(X_test,y_test)
eli5.show_weights(perm, feature_names = val_X.columns.tolist())"""

#%%
X_matrix = X.as_matrix()
y_matrix = y.as_matrix()
xgb = XGBClassifier()
xgb.fit(X_matrix, y_matrix)
perm_xgb = PermutationImportance(xgb).fit(X_matrix, y_matrix)
eli5.show_weights(perm_xgb, feature_names=list(X))
