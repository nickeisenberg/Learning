from sys import exit
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

gme_info = yf.Ticker('GME')

gme_df_2y = gme_info.history(period='2y',
                             interval='1h',
                             actions=False)

gme = gme_df_2y.reset_index()
gme.rename(columns={'index' : 'Date'}, inplace=True)

p_change = np.divide(gme['Close'].values, gme['Open'].values) - 1
gme['Percent_Change'] = p_change

print(gme.head(15))



