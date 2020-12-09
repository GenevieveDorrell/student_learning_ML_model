# -*- coding: utf-8 -*-
"""
@author: Czander

Basic ML implementation for 'Digit Recognizer' dataset.
No data wrangling at all, for 1000/42000 samples. 

Results (accuracy)
Logistic Regression:    91.56
Random Forest:          96.65
Decision Tree:          85.51
"""
import pandas as pd
import numpy as np
import time
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
#pytorch
import torch
import torch.nn as nn
import torch.nn.functional as F
#keras
#import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def eval(y_pred, y):
    count = 0
    for i,val in enumerate(y_pred):
        if val == y[i]:
            count += 1
    return round((count/len(y)) * 100, 2)


train_df = pd.read_csv('train.csv')
samp_df = train_df[:]       #to toggle sample size

#data with validation split -- ~ 80-20 split (training/validation)
train_set = samp_df[:33600].copy()
test_set = samp_df[33601:].copy().reset_index()
x = train_set.drop(columns=['label'])
y = train_set['label']
x_test = test_set.drop(columns=['index','label'])
y_test = test_set['label']

"""
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

#nueral network & deep learning

#intial attempt at a neural network??????????
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 10 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 10, 28)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 28 * 28, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1's .weight

input = train_df
out = net(input)
print(out)

net.zero_grad()
out.backward(torch.randn(1, 10))
"""
#keras
# Get the data as Numpy arrays
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

#print(x_test)
#print(y_test)

# Build a simple model
inputs = keras.Input(shape=(28, 28))
x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(10, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Train the model for 1 epoch from Numpy data
batch_size = 64
print("Fit on NumPy data")
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=1)

# Train the model for 1 epoch using a dataset
dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
print("Fit on Dataset")
history = model.fit(dataset, epochs=1)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
print("Generate predictions for 3 samples")
predictions = model.predict(x_test[:3], 3)
print("predictions shape:", predictions.shapes)
print(y_test[:3])