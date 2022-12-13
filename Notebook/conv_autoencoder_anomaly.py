import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler

np.random.seed(1)

time = np.linspace(0, 5, 5000)

def bump(x, s, plateau=False):
    if plateau:
        b = np.minimum(np.exp(- (x - s) ** 2 / .005) / np.sqrt(.005), 12)
    else:
        b = np.exp(- (x - s) ** 2 / .005) / np.sqrt(.005)
    return b

train_data = np.zeros(5000)
for i in np.arange(0, 5, 1 / 2)[1:]:
    train_data += bump(time, i)
# train_data /= max(train_data)
train_data += np.sqrt(1 / 5000) * np.random.normal(0, 1, size=5000)
train_data = train_data.reshape((-1, 1))

seq_len = 700

plt.plot(time[: seq_len], train_data[:seq_len])
plt.plot(time[seq_len:], train_data[seq_len:])
plt.show()

anomaly = bump(time, 3) / max(bump(time, 3))
anomaly = anomaly.reshape((-1, 1))
anomalous_data = train_data + 6 * anomaly

plt.plot(anomalous_data)
plt.show()

Mm = MinMaxScaler(feature_range=(0,1))

train_data = Mm.fit_transform(train_data)

train_data = train_data.reshape(-1)
plt.plot(train_data)
plt.show()

anomalous_data = Mm.transform(anomalous_data)

anomalous_data = anomalous_data.reshape(-1)
plt.plot(anomalous_data)
plt.show()

train_inputs = keras.utils.timeseries_dataset_from_array(
        data=train_data,
        targets=None,
        sequence_length=seq_len,
        batch_size=None)
train_inputs = np.array([np.array(inp) for inp in train_inputs])

print(train_inputs.shape)

train_inputs = train_inputs.reshape(
        train_inputs.shape[0], train_inputs.shape[1], 1)

print(train_inputs.shape)

# Convolution autoencoder
inputs = keras.Input(shape=(seq_len, 1))
x = keras.layers.Conv1D(
        filters=32, kernel_size=7, padding='same', strides=2,
        activation='relu')(inputs)
x = keras.layers.Dropout(.2)(x)
x = keras.layers.Conv1D(
        filters=16, kernel_size=7, padding='same', strides=2,
        activation='relu')(x)
x = keras.layers.Conv1DTranspose(
        filters=16, kernel_size=7, padding='same', strides=2,
        activation='relu')(x)
x = keras.layers.Dropout(.2)(x)
x = keras.layers.Conv1DTranspose(
        filters=32, kernel_size=7, padding='same', strides=2,
        activation='relu')(x)
outputs = keras.layers.Conv1DTranspose(
        filters=1, kernel_size=7, padding='same')(x)
model = keras.Model(inputs, outputs)
model.summary()

# compile the models
model.compile(optimizer='adam',
              loss='mae',
              metrics=['mae'])

model_dir = '/Users/nickeisenberg/GitRepos/Python_Notebook/Notebook/Models'

callbacks = [
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=5,
            mode='min'),
        keras.callbacks.ModelCheckpoint(
            filepath=f'{model_dir}/conv_anom.keras',
            monitor='val_mae',
            save_best_only=True)
        ]

# fit the convolutional and lstm autoencoder
history = model.fit(train_inputs, train_inputs,
                         batch_size=128,
                         validation_split=.1,
                         callbacks=callbacks,
                         epochs=50)

# loss for for models
train_loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(train_loss) + 1)
plt.plot(epochs, train_loss)
plt.plot(val_loss)
plt.show()

# predictions for both models on the train data
train_inputs_pred = model.predict(train_inputs)

# Test the predicted train inputs
plt.plot(train_inputs_pred[40])
plt.plot(train_inputs[40])
plt.show()

train_mae = np.mean(np.abs(train_inputs - train_inputs_pred), axis=1)
threshold = np.max(train_mae)

test_inputs = keras.utils.timeseries_dataset_from_array(
        data=anomalous_data,
        targets=None,
        sequence_length=seq_len,
        batch_size=None)

test_inputs = np.array([np.array(ti) for ti in test_inputs])

test_inputs = test_inputs.reshape(
        test_inputs.shape[0], test_inputs.shape[1], 1)

print(test_inputs.shape)

test_inputs_pred = model_conv.predict(test_inputs)

a_e = np.abs(test_inputs_pred - test_inputs)
test_mae = np.mean(a_e, axis=1)

start_ind = range(len(test_mae))

plt.plot(start_ind, test_mae)
plt.plot(start_ind, [threshold for i in start_ind])
plt.show()

anomalies_ind = []
anomalies_val = []
for i, mae in enumerate(test_mae.reshape(-1)):
    if mae > threshold_lstm:
        anomalies_ind.append(i)
        anomalies_val.append(anomalous_data[i])
anomalies_ind = np.array(anomalies_ind)
anomalies_val = np.array(anomalies_val)


plt.plot(time, anomalous_data)
plt.scatter(time[anomalies_ind], anomalies_val, c='red', s=10)
plt.title('predicted anomolies')
plt.show()
