import numpy as np
import matplotlib.pyplot as plt
from lstm_test import *
from lstm_funcs import *
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import load_model

data_test = data[1300:].reshape((-1,1))

Mm = MinMaxScaler(feature_range=(0,1))

inps_test = data[len(data) - len(data_test) - 50 : ].reshape((-1,1))
inps_test = Mm.fit_transform(inps_test)

X_test, _ = lstm_inp_out_generator(inps_test, lookback=50)

model = load_model('/Users/nickeisenberg/GitRepos/Python_Misc/MinimalWorkingExamples/Models/test_lstm')

pred = model.predict(X_test)
pred = Mm.inverse_transform(pred)

plt.subplot(121)
plt.plot(data_test, label='real')
plt.plot(pred, label='model.predict')
plt.legend()
plt.title('model.predict')

past = data[:1300]
future_preds = future_prediction(data=past, model=model, pred_amt=700, lookback=50)

plt.subplot(122)
plt.plot(data_test, label='real')
plt.plot(future_preds, label='preds')
plt.legend()
plt.title('future preditions only given the past')
plt.show()
