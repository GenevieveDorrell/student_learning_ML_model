import pandas as pd
import numpy as np
#keras
#dataset https://www.kaggle.com/crawford/emnist?select=emnist-balanced-train.csv
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# accuracy = 83.54
# Get the data as Numpy arrays
#train_set = np.genfromtxt('emnist-balanced-train.csv',delimiter=',')
train_set = pd.read_csv('emnist-balanced-train.csv',header=None)
x_train = train_set.drop(0, axis='columns')
y_train = train_set[0]
x_train = pd.DataFrame.to_numpy(x_train)
y_train = pd.DataFrame.to_numpy(y_train)

test_set = pd.read_csv('emnist-balanced-test.csv',header=None)
x_test = test_set.drop(0, axis='columns')
y_test = test_set[0]
x_test = pd.DataFrame.to_numpy(x_test)
y_test = pd.DataFrame.to_numpy(y_test)
x_test = x_test.reshape(-1, 28, 28, 1)
x_train = x_train.reshape(-1, 28, 28, 1)

pool_size = (2, 2)

# Build a simple model
inputs = keras.Input(shape=(28, 28, 1))
x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
#x = inputs.reshape(-1, 28, 28, 1)
x = layers.Convolution2D(25, 5, 5, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=pool_size)(x)
x = layers.Convolution2D(25, 5, 5, activation='relu')(x)
x = layers.MaxPooling2D(pool_size=pool_size)(x)
x = layers.Convolution2D(25, 4, 4, activation='relu')(x)
#x = layers.experimental.preprocessing.Rescaling(1.0 / 255)(inputs)
#reshape(-1, 28, 28, 1)
x = layers.Flatten()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dense(128, activation="relu")(x)
outputs = layers.Dense(47, activation="softmax")(x)
model = keras.Model(inputs, outputs)
model.summary()

# Compile the model
model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")

# Train the model for 1 epoch from Numpy data
batch_size = 64
print("Fit on NumPy data")
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=15)

# Train the model for 1 epoch using a dataset
#dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
#print("Fit on Dataset")
#history = model.fit(dataset, epochs=15)

# Evaluate the model on the test data using `evaluate`

#results = model.evaluate(x_test, y_test, batch_size=128)

#print("test loss, test acc:", results)

# Generate predictions (probabilities -- the output of the last layer)
# on new data using `predict`
#print("Generate predictions for 3 samples")
#predictions = model.predict(np.array(x_test[0]), 1)
#print(predictions)
#print("predictions shape:", predictions.shapes)
#print(y_test[:3])


def eval(y_pred, y):
    count = 0
    for i,row in enumerate(y_pred):
        row = list(row)
        val = row.index(max(row))
        count += int(val == y[i])
    return round((count/len(y)) * 100, 2)
print("Evaluate on test data")
pred = model.predict(x_test)
acc = eval(pred, y_test)
print(acc)