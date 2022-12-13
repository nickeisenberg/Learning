import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

np.random.seed(1)

time = np.linspace(0, 3, 1000)

def bump(x, s, plateau=False):
    if plateau:
        b = np.minimum(np.exp(- (x - s) ** 2 / .005) / np.sqrt(.005), 12)
    else:
        b = np.exp(- (x - s) ** 2 / .005) / np.sqrt(.005)
    return b

data = np.zeros(1000)
for i in np.arange(0, 3, 1 / 3)[1:]:
    if i == 1.:
        data += bump(time, 1, plateau=True)
    else:
        data += bump(time, i)

data /= max(data)
anomoly = bump(time, 2)
anomoly /= 25
data += anomoly

plt.plot(data)
plt.show()

seq_len=50
train_targets = keras.utils.timeseries_dataset_from_array(
        data=data,
        targets=None,
        sequence_length=seq_len,
        batch_size=None)

train_targets = np.array([np.array(tar) for tar in targets])

# train_inputs = []
# for tar in targets:
#     train_inputs.append(tar[::-1])
# train_inputs = np.array(train_inputs)

inputs = keras.Input(shape=(seq_len, 1))
x = keras.layers.LSTM(64, return_sequences=True)(inputs)
x = keras.layers.LSTM(32, return_sequences=True)(x)
outputs = keras.layers.TimeDistributed(keras.layers.Dense(1))(x)
model = keras.Model(inputs, outputs)
model.summary()

model.compile(optimizer='adam',
              loss='mae',
              metrics=['mae'])

history = model.fit(train_targets, train_targets,
                    epochs=10)

print(history.history.keys())

train_loss = history.history['loss']
train_mae = history.history['mae']

plt.plot(data)
plt.show()

# test_inputs = keras.utils.timeseries_dataset_from_array(
#         data=data,
#         targets=None,
#         sequence_length=seq_len,
#         batch_size=None)
# test_inputs = np.array([np.array(ti[::-1]) for ti in test_inputs])

preds = []
count = 1
for ti in :
    print(f'{count / len(test_inputs)}')
    preds.append(model.predict(ti.reshape(50, 1)).reshape(-1))
    count += 1

preds_data = np.array(preds)
print(preds_data.shape)

p = preds_data[::50].reshape(-1)
plt.plot(p)
plt.plot(data)
plt.show()

preds_mae = []
for td, pd in zip(test_inputs, preds_data):
    mae = np.mean(np.abs(td[::-1] - pd))
    preds_mae.append(mae)
preds_mae = np.array(preds_mae)

threshold = .28
time = range(len(preds_mae))

plt.clf()

plt.plot(time, [threshold for i in time])
plt.plot(time, preds_mae)
plt.show()
