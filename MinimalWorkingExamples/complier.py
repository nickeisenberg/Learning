import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
from copy import deepcopy
from sys import exit

url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/NSE-TATAGLOBAL.csv'
dataset_train = pd.read_csv(url)[::-1].reset_index(drop=True)
training_set = dataset_train.iloc[:, 1:2].values
print(training_set[:5])

dataset_train = dataset_train.set_index(dataset_train['Date'])
dataset_train = dataset_train.drop('Date', axis=1)
training_set = dataset_train['Open'].values
print(training_set[:5])

url = 'https://raw.githubusercontent.com/mwitiderrick/stockprice/master/tatatest.csv'
dataset_test = pd.read_csv(url)[::-1].reset_index(drop=True)
real_stock_price = dataset_test.iloc[:, 1:2].values

dataset_total = pd.concat((dataset_train, dataset_test), axis=0).reset_index(drop=True)
dataset_total = dataset_total.set_index(dataset_total['Date'])
dataset_total = dataset_total.drop('Date', axis=1)
print(dataset_train[:4])
print(dataset_train[-4:])
print('--------------------------------------------------')

tata_info = yf.Ticker('TATACONSUM.NS')
tata_df = tata_info.history(period='max',
                            interval='1d',
                            actions=False).reset_index(drop=False)
print('---')
print(tata_df['Open'].values.shape)
print('---')
date_start = dataset_train.index.values[0]
date_end = dataset_train.index.values[-1]
ind_start = tata_df.loc[tata_df['Date'] == date_start].index.values[0]
ind_end = tata_df.loc[tata_df['Date'] == date_end].index.values[0]

tata_dfsub = tata_df.iloc[ind_start:ind_end+1]
tata_dfsub = tata_dfsub.set_index('Date')

print(tata_dfsub[:4])
print(tata_dfsub[-4:])
