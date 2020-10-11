# -*- coding: utf-8 -*-
"""
Created on Fri Oct  9 16:24:15 2020

@author: Czander
"""

import pandas as pd
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier

def wrangle(df):
    try:
        df = df.drop(columns=['Name', 'Ticket', 'Cabin', 'Embarked'])       #dropping columns that will not be used in training
        df['Sex'] = df['Sex'].map({'female': 1, 'male': 0}).astype(int)     #converting 'Sex' to categorical format
    except: pass
    for i,val in enumerate(df['Age']):
        if np.isnan(val):
            df.at[i, 'Age'] = 21                    #converting NaN values in 'Age' to 21
    for i,val in enumerate(df['Fare']):
        if np.isnan(val):
            df.at[i, 'Fare'] = 7                    #converting NaN values in 'Fare' to 7
    return df

def eval(y_pred, y):
    count = 0
    for i,val in enumerate(y_pred):
        if val == y[i]:
            count += 1
        elif val == -1 and y[i] == 0:
            count += 1
    return round((count/len(y)) * 100, 2)


#Read in data
train_df = pd.read_csv('train.csv')
test_df = pd.read_csv('test.csv')

#Data Wrangling
train_df = wrangle(train_df)
test_df = wrangle(test_df)

'''
#data with no validation split -- with full training data (not used below)
x = train_df[['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = train_df['Survived']

test_set = test_df[['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
'''

#data with validation split -- ~ 80-20 split (training/validation)
train_set = train_df[:700].copy()
test_set = train_df[701:].copy().reset_index()
x = train_set[['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y = train_set['Survived']
x_test = test_set[['Pclass','Sex', 'Age', 'SibSp', 'Parch', 'Fare']]
y_test = test_set['Survived']


#SVC
classifier = SVC(kernel = 'rbf', verbose=True)
classifier.fit(x, y)

y_pred = classifier.predict(x_test)
svc_score = round(classifier.score(x, y) * 100, 2)
acc_svc = eval(y_pred, y_test.values)


#Logistic Regression
regressor = LogisticRegression(verbose=1)
regressor.fit(x, y)

y_rg = regressor.predict(x_test)
rg_score = round(regressor.score(x, y) * 100, 2)
acc_rg = eval(y_rg, y_test.values)


#Random Forest
rf_class = RandomForestClassifier()
rf_class.fit(x, y)

y_rf = rf_class.predict(x_test)
rf_score = round(rf_class.score(x, y) * 100, 2)
acc_rf = eval(y_rf, y_test.values)


#Decisison Tree
dtree = DecisionTreeClassifier()
dtree.fit(x, y)

y_dt = dtree.predict(x_test)
dt_score = round(dtree.score(x, y) * 100, 2)
acc_dt = eval(y_dt, y_test.values)