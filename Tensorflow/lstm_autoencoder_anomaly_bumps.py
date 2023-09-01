import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from plotly.subplots import make_subplots
 
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
train_data += np.sqrt(1 / 5000) * np.random.normal(0, 1, size=5000)
seq_len = 20

plt.plot(time[: seq_len], train_data[:seq_len])
plt.plot(time[seq_len:], train_data[seq_len:])
plt.show()

anomaly = np.zeros(5000)
anomaly += 5 * bump(time, 3) / max(bump(time, 3))
anomaly += 1 * bump(time, 4) / max(bump(time, 4))
anomaly -= 4 * bump(time, 1) / max(bump(time, 1))
anomaly -= 2 * bump(time, 2) / max(bump(time, 2))
anomaly += np.hstack((
    np.zeros(4600), np.array([1, 2, 3, 4, 5]), np.zeros(5000 - 4600 -5)))
anomaly -= np.hstack((
    np.zeros(420), np.array([2, 1, 2]), np.zeros(5000 - 420 - 3)))
anomalous_data = train_data + anomaly

fig = make_subplots(rows=2, cols=1,
                    subplot_titles=(
                        'training signal',
                        'anomalous signal'))
_ = fig.add_trace(go.Scatter(x=time, y=train_data,
                             line={'width': 3,
                                   'color': 'black'},
                             name='training signal'),
                  row=1, col=1)
_ = fig.add_trace(go.Scatter(x=time, y=anomalous_data,
                             line={'width': 3,
                                   'color': 'blue'},
                             name='anomalous signal'),
              row=2, col=1)
_ = fig.update_layout(yaxis1={'range': [0, 20]},
                      yaxis2={'range': [0, 20]})
fig.show()

train_inputs = keras.utils.timeseries_dataset_from_array(
        data=train_data,
        targets=None,
        sequence_length=seq_len,
        batch_size=None)
train_inputs = np.array([np.array(inp) for inp in train_inputs])
 
# LSTM autoencoder
inputs = keras.Input(shape=(seq_len,))
x = keras.layers.Reshape((seq_len, 1))(inputs)
x = keras.layers.LSTM(128, activation='relu', return_sequences=True)(x)
x = keras.layers.LSTM(64, activation='relu')(x)
x = keras.layers.RepeatVector(seq_len)(x)
x = keras.layers.LSTM(64, activation='relu', return_sequences=True)(x)
x = keras.layers.LSTM(128, activation='relu', return_sequences=True)(x)
x = keras.layers.TimeDistributed(keras.layers.Dense(1))(x)
outputs = keras.layers.Reshape((seq_len,))(x)
model = keras.Model(inputs, outputs)

model.summary()   

model.compile(optimizer='adam',
              loss='mae',
              metrics=['mae'])

model_dir = '/Users/nickeisenberg/GitRepos/Python_Notebook/Notebook/Models/'

callbacks = [
        keras.callbacks.ModelCheckpoint(
            filepath=f'{model_dir}lstm_anom_bumps.keras',
            monitor='val_mae',
            save_best_only=True)
        ]

history = model.fit(train_inputs, train_inputs,
                    batch_size=128,
                    validation_split=.1,
                    callbacks=callbacks,
                    epochs=200)

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

model = keras.models.load_model(
        model_dir + '/lstm_anom.keras')

train_preds = model.predict(train_inputs)

print(train_preds.shape)

pred_signal = np.hstack(
        train_preds[::seq_len])

print(pred_signal.shape)

# Encode and decode the input data
plt.plot(train_data, label='input data', linewidth=4)
plt.plot(pred_signal, label='encdoed and decoded\ninput data')
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
 
print(test_inputs.shape)
 
test_preds = model.predict(test_inputs)

print(test_preds.shape)

# visualize the encoding and decoding of the test_inputs
test_pred_signal = np.hstack(
        test_preds[::seq_len])

plt.plot(anomalous_data, label='test data')
plt.plot(test_pred_signal, label='encoded and decoded test data')
plt.legend(loc='upper left')
plt.title('encoded and decoded anomalous data')
plt.show()
 
a_e = np.abs(test_preds - test_inputs)
print(a_e.shape)

test_mae = np.mean(a_e, axis=1)

start_ind = np.array([*range(len(test_mae))])
 
plt.plot(start_ind, test_mae)
plt.plot(start_ind, [threshold for i in start_ind])
plt.show()

fig = make_subplots(rows=2, cols=1)
_ = fig.add_trace(go.Scatter(x=time, y=anomalous_data,
                             line={'width': 4},
                             name='anomalous signal'),
                  row=1, col=1)
_ = fig.add_trace(go.Scatter(x=time, y=test_pred_signal,
                             name='encoded and decoded anomalous signal',
                             line={'width': 2}),
                  row=1, col=1)
_ = fig.add_trace(go.Scatter(x=start_ind, y=test_mae,
                             line={'width': 2},
                             name='anomalous signal MAE'),
                  row=2, col=1)
_ = fig.add_trace(go.Scatter(
    x=start_ind, y=np.array([threshold for i in start_ind]),
    line={'width': 3},
    name='MAE threshold'),
                  row=2, col=1)
_ = fig.update_layout(
        title={'text': 'Detection of anomlies using the test MAE',
               'x': .5})
fig.show()

anomalies_ind = []
for i, mae in enumerate(test_mae):
    if mae > threshold:
        if len(anomalies_ind) == 0:
            anomalies_ind.append(i)
        elif i > anomalies_ind[-1] - seq_len / 2:
            anomalies_ind.append(i)
anomalies_ind = np.array(anomalies_ind)

detected_anomalies = []
for i in anomalies_ind:
    if i == anomalies_ind[0]:
        detected_anomalies.append(
                go.Scatter(x=time[i: i + seq_len],
                           y=anomalous_data[i: i + seq_len],
                           line={'color': 'red',
                                 'width': 2},
                           name='detected anomaly')
                )
    else:
        detected_anomalies.append(
                go.Scatter(x=time[i: i + seq_len],
                           y=anomalous_data[i: i + seq_len],
                           line={'color': 'red',
                                 'width': 2},
                           showlegend=False)
                )
fig = make_subplots(rows=2, cols=1)
_ = fig.add_trace(
        go.Scatter(x=time, y=train_data,
                   line={'color': 'black',
                         'width': 3},
                   name='training signal'),
        row=1, col=1
        )
_ = fig.add_trace(
        go.Scatter(x=time, y=anomalous_data,
                   line={'width': 3,
                         'color': 'blue'},
                   name='anomalous signal'),
        row=2, col=1
        )
for plot in detected_anomalies:
    _ = fig.add_trace(
            go.Scatter(plot),
            row=2, col=1
            )
_ = fig.update_layout(yaxis1={'range': [0, 20]},
                      yaxis2={'range': [0, 20]},
                      title={'text': 'Detected anomalies',
                             'x': .5})
fig.show()


