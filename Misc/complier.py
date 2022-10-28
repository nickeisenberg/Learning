import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from copy import deepcopy

url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
dataset_train = pd.read_csv(url)
training_set = dataset_train.iloc[:, 1:2].values
# print(dataset_train.head())
print(dataset_train.head(-1))

print('--------------------------------------------------')

url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/tatatest.csv'
dataset_test = pd.read_csv(url)
real_stock_price = dataset_test.iloc[:, 1:2].values
# print(dataset_test.head())
print(dataset_test)

print('--------------------------------------------------')

dataset_total = pd.concat((dataset_train, dataset_test), axis=0)
print(dataset_total.head(-1))

gme_info = yf.Ticker('GME')
gme_df = gme_info.history(period='2y',
                          interval='1h',
                          actions=False)

gme_open = gme_df.iloc[:, 1:2].values
gme_train = gme_open[ : 2500]

X_train = []
y_train = []

for i in range(60, len(gme_train)):
    X_train.append(gme_train[i-60: i, 0])
    y_train.append(gme_train[i, 0])

X_train, y_train = np.array(X_train), np.array(y_train)

x = np.array([1, 1, 1]).reshape(3, 1)
x1 = deepcopy(x)
l = []
l.append(x[:,0])
l.append(x1[:,0])
l = np.array(l)


