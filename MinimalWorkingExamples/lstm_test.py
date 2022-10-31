import numpy as np
import matplotlib.pyplot as plt
from sys import exit
from lstm_funcs import *

time = np.linspace(0, 1, 2000)

freq = [5, 4, 11, 7]
data = []
for f in freq:
    data_ = np.sin(2 * np.pi * f * time)
    data.append(data_)
data = np.array(data)
data = np.sum(data, axis=0)

data_train = data[:1300]
data_train = data_train.reshape((-1,1))
from sklearn.preprocessing import MinMaxScaler
Mm = MinMaxScaler(feature_range=(0,1))
data_train_scaled = Mm.fit_transform(data_train)

# def lstm_inp_out_generator(data=None, lookback=None, output_len=1):
#     if not isinstance(lookback, int):
#         print('enter a integer for lookback')
#         return None
# 
#     if lookback > len(data):
#         print('lookback must be less than the length of the data')
#         return None
# 
#     inps = []
#     outs = []
# 
#     for i in range(lookback, len(data)):
#         inps.append(data[i - lookback : i , 0])
#         outs.append(data[i, 0])
# 
#     inps, outs = np.array(inps), np.array(outs)
#     inps, outs = inps.reshape((inps.shape[0], inps.shape[1], 1)), outs.reshape((-1, 1))
# 
#     return inps, outs

# test = np.array([1, 2, 3, 4, 5]).reshape((-1, 1))
# a, b = lstm_inp_out_generator(data=test, lookback=3)
# print(a)
# print(b)

if __name__ == '__main__':
    X_train, y_train = lstm_inp_out_generator(data=data_train_scaled, lookback=50)

    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import LSTM, Dropout, Dense

    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(X_train.shape[1], 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.fit(X_train, y_train, epochs=50, batch_size=32)

    model.save('/Users/nickeisenberg/GitRepos/Python_Misc/MinimalWorkingExamples/Models/test_lstm')
