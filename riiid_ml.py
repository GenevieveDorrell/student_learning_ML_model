# -*- coding: utf-8 -*-
"""
@author: Czander

Basic implementation of ML to the Riiid dataset.
Just to clarify, the methods implemented below do not actually work towards
the knowledge tracing problem. I'm not really familiar with this problem
and how exactly to structure the data for it. Seems to be an advanced
ML problem.

"""

import pandas as pd
import numpy as np
import time
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

def count_users(df):
    user_count = dict()
    for user in df['user_id']:
        if user in user_count:
            user_count[user] += 1
        else:
            user_count[user] = 1
    return user_count

def eval(y_pred, y):
    count = 0
    for i,val in enumerate(y_pred):
        if val == y[i]:
            count += 1
        elif val == -1 and y[i] == 0:
            count += 1
    return count/len(y)

#Use this to load data from 'train.csv' and then save as a pickle file
train_df = pd.DataFrame()
for chunk in pd.read_csv('train.csv', chunksize=1000000, low_memory=False,
                        dtype={'row_id': 'int64', 'timestamp': 'int64', 'user_id': 
                       'int32', 'content_id': 'int16', 'content_type_id': 'int8',
                       'task_container_id': 'int16', 'user_answer': 'int8', 
                       'answered_correctly': 'int8', 'prior_question_elapsed_time': 
                    'float32','prior_question_had_explanation': 'boolean',}):
    train_df = pd.concat([train_df, chunk], ignore_index=True)

train_df.to_pickle("train.pkl")

train_df = pd.read_pickle("train.pkl")            #load time: ~36s
total_users = count_users(train_df)               #counting number of students in the dataset
samp_df = train_df[:10000].copy()                 #using a chunk of the data for tinkering

## Data Wrangling
#converting NaN values in 'prior_question_elpased_time' to 0
for i,val in enumerate(samp_df['prior_question_elapsed_time']):
    if val > 0: pass
    else:
        samp_df.at[i,'prior_question_elapsed_time'] = 0

#converting <NA> values in 'prior_question_had_explanation' to False
for i,val in enumerate(samp_df['prior_question_had_explanation']):
    if pd.isna(val):
        samp_df.at[i, 'prior_question_had_explanation'] = False

#remove -1 values in 'answered correctly' (lectures)
droplist = []
for i,val in enumerate(samp_df['answered_correctly']):
    if val < 0:
        droplist.append(i)
samp_df = samp_df.drop(droplist)


#Setting up training set and test set
df = samp_df[:9000]
test_df = samp_df[9001:].copy().reset_index()

x = df[['content_id','task_container_id','prior_question_elapsed_time',
                       'prior_question_had_explanation']]
y = df['answered_correctly']

test_set = test_df[['content_id','task_container_id','prior_question_elapsed_time',
                    'prior_question_had_explanation']]


#Support Vector Machine classification
classifier = SVC(kernel = 'rbf', verbose=True)          #initializing classification model
classifier.fit(x, y)                                    #training classifier

y_pred = classifier.predict(test_set)                   #using classifier to predict on test_set
acc_svc = eval(y_pred, test_df['answered_correctly'])   #evaluating prediction results


#Logistic Regression
regressor = LogisticRegression(verbose=1)               #initializing regression model
regressor.fit(x, y)                                     #training regressor

y_pred_lr = regressor.predict(test_set)                 #using regressor to predict on test_set
acc_lr = eval(y_pred_lr, test_df['answered_correctly']) #evaluating prediction results
