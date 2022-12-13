import tensorflow as tf
from tensorflow import keras
import numpy as np

class custom_layer(keras.layers.Layer):

    def __init__(self, in_dim=32, out_dim=32):
        super().__init__()
        self.w = np.random.normal(0, 1, size=(in_dim, out_dim))
        self.b = np.random.normal(0, 1, size=out_dim)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b

train_x = []
train_y = []
for i in range(100):
    train_x.append(np.random.normal(0, 1, 32))
    train_y.append(np.random.normal(0, 1, 1)[0])
train_x = np.array(train_x)
train_y = np.array(train_y)

inputs = keras.Input(shape=(32,))
x = custom_layer()(inputs)
outputs = keras.layers.Dense(1)(x)
model = keras.Model(inputs, outputs)
model.summary()

model.compile(optimizer='adam', loss='mae', metrics=['mae'])

history = model.fit(train_x,
                    train_y,
                    epochs=10)

