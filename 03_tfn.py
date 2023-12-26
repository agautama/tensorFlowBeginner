import os
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
import numpy as np
from matplotlib import pyplot as plt

print(tf.__version__)

mnist = keras.datasets.mnist

(x_train, y_train), (x_test, y_test) = mnist.load_data()

print(x_train.shape, y_train.shape)
print(x_test.shape, y_test.shape)

# normalize divide values by 255

x_train, x_test = x_train / 255.0, x_test / 255.0

# for i in range(6):
#     plt.subplot(2,3,i+1)
#     plt.imshow(x_train[i], cmap = 'gray')
# plt.show()

# model
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation = 'relu'),
    keras.layers.Dense(10)
])

model.summary()

# Loss and optimizer

loss = keras.losses.SparseCategoricalCross
