import numpy as np
from sklearn.preprocessing import MinMaxScaler

def lstm_inp_out_generator(data=None, lookback=None, output_len=1):

    if not isinstance(lookback, int):
        print('enter a integer for lookback')
        return None

    if lookback > len(data):
        print('lookback must be less than the length of the data')
        return None

    inps = []
    outs = []

    for i in range(lookback, len(data)):
        inps.append(data[i - lookback : i , 0])
        outs.append(data[i, 0])

    inps, outs = np.array(inps), np.array(outs)
    inps, outs = inps.reshape((inps.shape[0], inps.shape[1], 1)), outs.reshape((-1, 1))

    return inps, outs

def future_prediction(data=None, model=None, pred_amt=None, lookback=None):
    if len(data) < lookback:
        print('To big of a lookback')
        return None

    Mm = MinMaxScaler(feature_range=(0,1))
    data_scaled = Mm.fit_transform(data.reshape((-1,1))).T[0]

    pred_vals = []
    for i in range(pred_amt):
        if -lookback + i <= -1:
            data_lb = -lookback + i
            pred_lb = i
            inp_data = np.array(data_scaled[data_lb :])
            inp_pred = np.array(pred_vals[ : i])
            inp = np.hstack((inp_data, inp_pred))
            # print(inp_data)
            # print(inp_pred)
            # print(inp)
            # print(inp.reshape((1, len(inp), 1)))
            pred = model.predict(inp.reshape((1, len(inp), 1)))
            pred_vals.append(pred[0][0])

        if -lookback + i >= 0:
            inp = np.array(pred_vals[i - lookback : i])
            pred = model.predict(inp.reshape((1, len(inp), 1)))
            pred_vals.append(pred[0][0])

    pred_vals = Mm.inverse_transform(np.array(pred_vals).reshape((-1, 1)))
    return pred_vals.reshape(-1)

