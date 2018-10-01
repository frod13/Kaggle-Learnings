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
eli5.explain_weights_df(perm_xgb, feature_names=list(X))

html = permutation_html.data
with open('output/permutation.html', 'w') as f:
    f.write(html)
    
#%% basic implementation of partial dependence plots
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

#can't we use non-numerical data for PDEs? test below
fifa_data = pd.read_csv('input/FIFA 2018 Statistics.csv')
feature_names = [i for i in fifa_data.columns if fifa_data[i].dtype in [np.int64]]
X = fifa_data[feature_names]
y = fifa_data['Man of the Match']
train_X, test_X, train_y, test_y = train_test_split(X, y, random_state=1)
tree_model = DecisionTreeClassifier(random_state=0, max_depth=5, min_samples_split=5).fit(train_X, train_y)

#%%print decision tree representation
from sklearn import tree
import graphviz
from IPython.display import Image

tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=feature_names,filled=True,rounded=True)
graphviz.Source(tree_graph)

#%% simple partial dependence plot

from matplotlib import pyplot as plt
from pdpbox import pdp, get_dataset, info_plots

# Create the data that we will plot
pdp_goals = pdp.pdp_isolate(model=tree_model, dataset=test_X, model_features=feature_names, feature='Goal Scored')

# plot it
_ = pdp.pdp_plot(pdp_goals, 'Goal Scored')
plt.show()

feature_to_plot = 'Distance Covered (Kms)'
feature_to_plot = 'Attempts'
pdp_dist = pdp.pdp_isolate(model=tree_model, dataset=test_X, model_features=feature_names, feature=feature_to_plot)
_ = pdp.pdp_plot(pdp_dist, feature_to_plot)
#add carpet or distribution plot to show quantity of observations
plt.show()

#%% same analysis with random forest

# Build Random Forest model
rf_model = RandomForestClassifier(random_state=0).fit(train_X, train_y)

for feature in feature_names:
    pdp_dist = pdp.pdp_isolate(model=rf_model, dataset=test_X, model_features=feature_names, feature=feature)
    pdp.pdp_plot(pdp_dist, feature)
    plt.show()
    
tree_graph = tree.export_graphviz(tree_model, out_file=None, feature_names=feature_names,filled=True,rounded=True)
graphviz.Source(tree_graph)

#%%2-D plot
# Similar to previous PDP plot except we use pdp_interact instead of pdp_isolate and pdp_interact_plot instead of pdp_isolate_plot
features_to_plot = ['Goal Scored', 'Distance Covered (Kms)']
features_to_plot = ['Goal Scored', 'Off-Target']
inter1  =  pdp.pdp_interact(model=rf_model, dataset=test_X, model_features=feature_names, features=features_to_plot)

pdp.pdp_interact_plot(pdp_interact_out=inter1, feature_names=features_to_plot, plot_type='contour')
plt.show()


#%% practice rf and xgboost classifiers with permutation importance and partial dependence plots

titanic_train = pd.read_csv('input/titanic_train.csv')
titanic_test = pd.read_csv('input/titanic_test.csv')

titanic_train.describe()
titanic_train.isnull().sum()
#177 missing values for Age, 687 missing values for Cabin, 2 missing values for Embarked

#%% train a simple random forest model regressor to input missing values for age. we drop Cabin and Embarked, we use the mode.
columns_impute = titanic_train.columns.tolist()
for x in ['Survived','Cabin','PassengerId','Name','Ticket']:
    try:
        columns_impute.remove(x)
    except:
        print('column',x,'not in list!')

titanic_train_impute = titanic_train[columns_impute]
titanic_test_impute = titanic_test[columns_impute]

#%%
titanic = titanic_train_impute.append(titanic_test_impute).reset_index(drop=True)
titanic['Embarked'] = titanic['Embarked'].fillna(titanic.Embarked.mode().iloc[0])
titanic[['Fare']].fillna(titanic.Fare.median(),inplace=True)

#one hot encode embarked column. should also do this before we decide to impute ages!!
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

#WRITE A FUNCTION TO CONVERT ALL STRING CATEGORIES TO One-hot Encode
for x in titanic.columns:
    if titanic[x].dtype not in ['float64','int64']:
              

#two-step process to onehotencode string columns
def ohe_encode(df,column):
    label_encoder = LabelEncoder()
    embark_ohe = OneHotEncoder(sparse=False)
    titanic['Embark_encoded'] = label_encoder.fit_transform(titanic.Embarked)
    embark_ohe_fit = embark_ohe.fit_transform(titanic.Embark_encoded.values.reshape(-1,1))
    dfOneHot = pd.DataFrame(embark_ohe_fit,columns = ["Embarked_"+label_encoder.inverse_transform(int(i)) for i in range(embark_ohe_fit.shape[1])])
    titanic = pd.concat([titanic, dfOneHot], axis=1)
    _ = titanic.pop('Embarked')
    _ = titanic.pop('Embark_encoded')

age_null = titanic[titanic['Age'].isnull()].reset_index()
age_notnull = titanic[~titanic['Age'].isnull()].reset_index()

from sklearn.ensemble import RandomForestRegressor

#apply to each row with a NaN value in age a prediction with the model
rf = RandomForestRegressor()
no_age = titanic.columns.tolist()
no_age.remove('Age')
rf.fit(age_notnull[no_age].values,age_notnull['Age'].values)
