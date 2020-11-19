import pandas as pd
import numpy as np
#keras
#accuracy 97.84
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
# Get the data as Numpy arrays
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()
#print(keras.datasets.mnist)
#print(x_train)
#print(y_train)

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
history = model.fit(dataset, epochs=20)

# Evaluate the model on the test data using `evaluate`
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

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
  
pred = model.predict(x_test)
acc = eval(pred, y_test)
print(acc)
