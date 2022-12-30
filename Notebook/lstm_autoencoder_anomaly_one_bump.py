import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
from copy import deepcopy
from keras.utils import timeseries_dataset_from_array 
from keras import layers, Input, Model, callbacks, models

# function to generate bumps
def similar_bumps(center, spread, height):
    partition = center
    # center += np.random.normal(loc=0, scale=center * .05, size=1)[0]
    spread += np.random.normal(loc=0, scale=spread * .1, size=1)[0]
    height += np.random.normal(loc=0, scale=height * .1, size=1)[0]
    bump = norm.pdf(
            np.linspace(0, 2 * partition, 2 * partition + 1),
            center,
            spread)
    bump /= np.max(bump)
    bump *= height
    return bump + .05 * np.random.normal(0, 1, 2 * partition + 1)
#--------------------------------------------------

# generate the bumps
np.random.seed(127)

good_bumps = np.array([
    similar_bumps(200, 25, 6) for i in range(100)])

outlier_bumps = np.array(
        [similar_bumps(200, 15, 10) for i in range(3)] +
        [similar_bumps(200, 15, 2) for i in range(2)])

bumps = np.vstack((good_bumps, outlier_bumps))

avg_bump = np.mean(bumps, axis=0)

time = np.linspace(0, len(avg_bump) - 1, len(avg_bump))

bump_plots = [
        go.Scatter(x=time, y=bump, showlegend=False)
        for bump in good_bumps] + [
                go.Scatter(x=time, y=bump)
                for bump in outlier_bumps]
bump_plots.append(go.Scatter(x=time, y=avg_bump,
                             line={'color': 'black',
                                   'width': 4},
                             name='average of the bumps'))
fig = go.Figure(bump_plots)
_ = fig.update_layout(
        title={'text': 'Bumps with labeled anomalies',
               'x': .5})
fig.show()
#--------------------------------------------------

# Create the training set based on the average of the bumps

train_bump = deepcopy(avg_bump[200 - 80: 200 + 80])

seq_len = 10

train_inputs = timeseries_dataset_from_array(
        data=train_bump,
        targets=None,
        sequence_length=seq_len,
        batch_size=None)

train_inputs = np.array([
    np.array(inp) for inp in train_inputs])

print(train_inputs.shape)
#--------------------------------------------------

# define the model
inputs = Input(shape=(seq_len,))
x = layers.Reshape((seq_len, 1))(inputs)
x = layers.LSTM(64, activation='relu', return_sequences=True)(x)
x = layers.LSTM(32, activation='relu', return_sequences=False)(x)
x = layers.RepeatVector(seq_len)(x)
x = layers.LSTM(32, activation='relu', return_sequences=True)(x)
x = layers.LSTM(64, activation='relu', return_sequences=True)(x)
x = layers.TimeDistributed(layers.Dense(1))(x)
outputs = layers.Reshape((seq_len,))(x)
model = Model(inputs, outputs)

model.summary()

model.compile(optimizer='adam',
              loss='mae',
              metrics=['mae'])
#--------------------------------------------------

# train the model
model_dir = '/Users/nickeisenberg/GitRepos/Python_Notebook/Notebook/Models/'

callbacks = [
        callbacks.ModelCheckpoint(
        filepath=model_dir + 'one_bump_anom.keras',
        monitor='val_mae',
        save_best_only=True)
        ]

history = model.fit(train_inputs, train_inputs,
                    batch_size=16,
                    validation_split=.1,
                    callbacks=callbacks,
                    epochs=200)
#--------------------------------------------------

# visualize the mae of the model
train_mae = history.history['mae']
val_mae = history.history['val_mae']
epochs = range(1, len(train_mae) + 1)
plt.plot(epochs, train_mae, label='train mae')
plt.plot(epochs, val_mae, label='val_mae')
plt.legend()
plt.show()
#--------------------------------------------------

# load the best model and see how it did on the training data
model = models.load_model(model_dir + 'one_bump_anom.keras')

train_preds = model.predict(train_inputs)

print(train_inputs.shape)
print(train_preds.shape)

train_pred_signal = np.hstack(
        [pred for pred in train_preds[::seq_len]]
        )

plt.plot(train_pred_signal)
plt.plot(train_bump)
plt.show()      
#--------------------------------------------------

# calculate the max mae to use as the anomoly threshhold
train_mae = np.mean(np.abs(train_inputs - train_preds), axis=1)
print(train_mae.shape)

threshold = np.max(train_mae)

# test the model on bumps[103] which should be detected as an anomoly
test_bump = deepcopy(bumps[103][200 -80: 200 + 80])

plt.plot(test_bump)
plt.plot(train_bump)
plt.show()

test_inputs = timeseries_dataset_from_array(
        data=test_bump,
        targets=None,
        sequence_length=seq_len,
        batch_size=None)

test_inputs = np.array([
    np.array(inp) for inp in test_inputs])

test_pred = model.predict(test_inputs)

test_pred_mae = np.mean(np.abs(test_pred - test_inputs), axis=1)

print(test_pred_mae.shape)
plt.plot(test_pred_mae)
plt.plot([threshold for i in range(len(test_pred_mae))])
plt.show()

count = 0
anomaly_detection_inds = []
for act, pred in zip(test_inputs, test_pred):
    mae = np.max(np.mean(np.abs(act - pred)))
    if mae >= threshold:
        if len(anomaly_detection_inds) == 0:
            anomaly_detection_inds.append(count)
        elif count > anomaly_detection_inds[-1] - seq_len / 2:
            anomaly_detection_inds.append(count)
    count += 1
#--------------------------------------------------
    
# See how the detection did
model_time = time[120: 280]

test_pred_bump = np.hstack(
        [pred for pred in test_pred[::seq_len]])

plt.plot(model_time, test_bump, c='green')
plt.plot(model_time, test_pred_bump c='blue')
for ind in anomaly_detection_inds:
    plt.plot(model_time[ind: ind + seq_len],
             test_pred_bump[ind: ind + seq_len],
             c='red')
plt.show()

fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=(
            'anomolus bump vs training bump',
            'detecting for an anomoly',)
        )
_ = fig.add_trace(
        go.Scatter(x=domain, y=test_bump,
                   name='trace 103',
                   line={'color': 'darkslategrey',
                         'width': 5}),
        row=1, col=2)
_ = fig.add_trace(
        go.Scatter(x=domain, y=test_pred_bump,
                   name='encoded and decoded anomolus bump',
                   line={'color': 'orange'}),
        row=1, col=2)
for anom in anomoly_detection_inds:
    _ = fig.add_trace(
            go.Scatter(x=domain[anom: anom + 10],
                       y=test_pred_bump[anom: anom + 10],
                       name='deteced anomoly',
                       mode='lines',
                       line={'color': 'red',
                             'width': 8}),
            row=1, col=2)
_ = fig.add_trace(
        go.Scatter(x=np.arange(120, 280, 1),
                   y=test_bump,
                   showlegend=False,
                   line={'color': 'darkslategrey',
                         'width': 5}),
        row=1, col=1)
_ = fig.add_trace(
        go.Scatter(x=np.arange(120, 280, 1), y=train_bump,
                   line={'color': 'black'},
                   name='average of the bumps'),
        row=1, col=1)
fig.show()
