import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from datetime import datetime

time = np.linspace(0, 10, 10000)
data = np.sin(2 * np.pi * 4 * time) + 3 * np.cos(2 * np.pi * 3 * time)
noise = np.random.normal(0, np.sqrt(10 / 1000), size=10000)

data += noise

plt.plot(data)
plt.show()

seq_len = 150
seq_len_out = 10

targets = keras.utils.timeseries_dataset_from_array(
        data=data[seq_len:],
        targets=None,
        sequence_length=seq_len_out,
        batch_size=None)

targets = np.array([np.array(tar) for tar in targets])

dataset = keras.utils.timeseries_dataset_from_array(
        data=data[: -seq_len_out],
        targets=targets,
        sequence_length=seq_len,
        batch_size=128)
 
inputs = keras.Input(shape=(150,1))
x = keras.layers.LSTM(64, recurrent_dropout=.2, return_sequences=True)(inputs)
x = keras.layers.LSTM(32, recurrent_dropout=.2, return_sequences=False)(x)
outputs = keras.layers.Dense(seq_len_out)(x)
model = keras.Model(inputs, outputs)
model.summary()

model.compile(optimizer='adam',
              loss='mae',
              metrics=['mae'])

history = model.fit(dataset,
                    epochs=10)

path = '/Users/nickeisenberg/GitRepos/Python_Notebook/Notebook/Models'
name = 'lstm_sin_cos.keras'
model.save(f'{path}/{name}')

model = keras.models.load_model(f'{path}/{name}')

test_time = np.linspace(1, 11, 10000)
test_data = np.sin(2 * np.pi * 4 * time) + 3 * np.cos(2 * np.pi * 3 * time)

test_dataset = keras.utils.timeseries_dataset_from_array(data=test_data,
                                                         targets=data[seq_len:],
                                                         sequence_length=seq_len)

model.evaluate(test_dataset)

start = test_data[: 150]

plt.plot(start)
plt.show()

preds = np.zeros(10000)
preds[: 150] = start

count = 1
stop = 100
for i in range(10000 - 150):
    print(f'completion percent: {100 * count / stop}')
    preds[150 + seq_len_out * i: 160 + seq_len_out * i] = model.predict(
            preds[seq_len_out * i: 150 + seq_len_out * i].reshape(1, 150, 1))
    if count == stop:
        break
    count += 1


stopped_test_time = test_time[: 150 + stop * seq_len_out]
plt.plot(stopped_test_time[: 150], preds[: 150], label='inital_input')
plt.plot(stopped_test_time[150: ],
         preds[150: 150 + stop * seq_len_out],
         label='recursively predicted outputs')

actual_dotted = test_data[150: 150 + stop * seq_len_out: 3]
stopped_test_time_dotted = stopped_test_time[150: 150 + stop * seq_len_out: 3]
plt.scatter(stopped_test_time_dotted, actual_dotted, s=10,
            label='actual outputs')

plt.title('A LSTM model trained to predict f(x) = Sin(x)\n\
          100 recursive preditions of length=100 simulated')
plt.legend(loc='lower right')
plt.xlim(0.9, 3)
plt.show()
