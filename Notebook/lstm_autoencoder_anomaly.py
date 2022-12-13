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
 
seq_len = 20

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

# LSTM autoencoder
inputs = keras.Input(shape=(seq_len, 1))
x = keras.layers.LSTM(128, activation='relu',
                      return_sequences=True)(inputs)
x = keras.layers.LSTM(64, activation='relu')(x)
x = keras.layers.RepeatVector(seq_len)(x)
x = keras.layers.LSTM(64, activation='relu', return_sequences=True)(x)
x = keras.layers.LSTM(128, activation='relu', return_sequences=True)(x)
outputs = keras.layers.TimeDistributed(keras.layers.Dense(1))(x)
model = keras.Model(inputs, outputs)
model.summary()   

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
            filepath=f'{model_dir}/lstm_anom_2.keras',
            monitor='val_mae',
            save_best_only=True)
        ]

history = model.fit(train_inputs, train_inputs,
                    batch_size=128,
                    validation_split=.1,
                    callbacks=callbacks,
                    epochs=50)

train_mae = history.history['mae']
val_mae = history.history['val_mae']
epochs = range(1, len(train_mae) + 1)
plt.plot(epochs, train_mae, label='train mae')
plt.plot(epochs, val_mae, label='val mae')
plt.legend()
plt.title('LSTM autoencoder train and validation mae')
plt.xlabel('epochs')
plt.ylabel('mae')
plt.show()

train_preds = model.predict(train_inputs)

print(train_preds.shape)

# Encode and decode the input data
pred_inp_data = train_preds.reshape(
        train_preds.shape[0], train_preds.shape[1]
        )[::20].reshape(-1)

plt.plot(train_data, label='input data')
plt.plot(pred_inp_data, label='encdoed and decoded\ninput data')
plt.legend(loc='upper left')
plt.title('The encoded and decoded input data')
plt.show()

train_mae = np.mean(np.abs(train_inputs - train_preds), axis=1)
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
 
test_preds = model.predict(test_inputs)

print(test_preds.shape)

# visualize the encoding and decoding of the test_inputs
pred_test_data = test_preds.reshape(
        test_preds.shape[0], test_preds.shape[1]
        )[::20].reshape(-1)

plt.plot(anomalous_data, label='test data')
plt.plot(pred_test_data, label='encoded and decoded test data')
plt.legend(loc='upper left')
plt.title('encoded and decoded anomalous data')
plt.show()
 
a_e = np.abs(test_preds - test_inputs)
print(a_e.shape)

test_mae = np.mean(a_e, axis=1)
print(test_mae.shape)

start_ind = range(len(test_mae))
 
plt.plot(start_ind, test_mae)
plt.plot(start_ind, [threshold for i in start_ind])
plt.show()

anomalies_ind_total = []
for i, mae in enumerate(test_mae.reshape(-1)):
    if mae > threshold:
        anomalies_ind_total.append(i)
anomalies_ind_total = np.array(anomalies_ind_total)

# Each anomaly index represents 20 timesteps in the future.
# In other words, if i in anomalies_ind, then that means [i: i + 20]
# is anomalous.
# Go through and remove redundant anomalies
anomalies_ind = []
for i in anomalies_ind_total:
    if len(anomalies_ind) == 0:
        anomalies_ind.append(i)
        continue
    if i > anomalies_ind[-1] + 20:
        anomalies_ind.append(i)
anomalies_ind = np.array(anomalies_ind)

print(anomalies_ind.shape)

print(anomalies_ind)

anomalies_val = []
for i in anomalies_ind:
    val = anomalous_data[i: i + 20]
    anomalies_val.append(val)
anomalies_val = np.array(anomalies_val)

print(anomalies_val.shape)

print(anomalies_val[0].shape)

# plot the anomalies
plt.plot(time, anomalous_data)
for i in anomalies_ind:
    plt.scatter(time[np.arange(i, i + 20, 1)],
                anomalous_data[i: i + 20],
                c='red', s=10)
plt.title('encoding and decoding of anomalous data')
plt.show()

   
 
