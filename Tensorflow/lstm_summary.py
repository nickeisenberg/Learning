from tensorflow import keras

inputs = keras.Input(shape=(100,5))
x = keras.layers.LSTM(128, return_sequences=True)(inputs)
x = keras.layers.LSTM(64, return_sequences=True)(x)
outputs = keras.layers.TimeDistributed(keras.layers.Dense(5))(x)

model = keras.Model(inputs, outputs)
model.summary()

