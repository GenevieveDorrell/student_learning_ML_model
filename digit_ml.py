# -*- coding: utf-8 -*-
"""

@author: Czander
"""

import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier


def eval(y_pred, y):
    count = 0
    for i,val in enumerate(y_pred):
        if val == y[i]:
            count += 1
        elif val == -1 and y[i] == 0:
            count += 1
    return round((count/len(y)) * 100, 2)


train_df = pd.read_csv('train.csv')

samp_df = train_df[:1000]

#data with validation split -- ~ 80-20 split (training/validation)
train_set = samp_df[:800].copy()
test_set = samp_df[801:].copy().reset_index()
x = train_set.drop(columns=['label'])
y = train_set['label']
x_test = test_set.drop(columns=['index','label'])
y_test = test_set['label']


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