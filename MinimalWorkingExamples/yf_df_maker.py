from sys import exit
import pandas as pd
import yfinance as yf

gme_info = yf.Ticker('GME')

gme_df_2y = gme_info.history(period='2y',
                               interval='1h',
                               actions=False)

gme_2y = pd.DataFrame(gme_df_2y['Open'], columns=['Open'])

gme_2y.to_csv('/Users/nickeisenberg/GitRepos/Python_Misc/MinimalWorkingExamples/DataSets/gme_11_3_22.csv')

