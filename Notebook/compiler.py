import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import yfinance as yf
from datetime import datetime
from scipy.stats import norm
import matplotlib.pyplot as plt

time = np.linspace(0, 1, 1000)
delta = time[1] - time[0]

b_motion = np.cumsum(
        np.sqrt(delta) * np.random.normal(0, 1, 1000)
        )

b_motion = np.hstack((0, b_motion))

fig = go.Figure(
        [go.Scatter(
            x=time,
            y=b_motion)
         ]
        )
fig.show()

b_motion_flucs = pd.DataFrame(
        data=np.diff(b_motion), columns=['flucs'])

fig = px.histogram(
        data_frame=b_motion_flucs,
        x='flucs',
        nbins=20)
fig.show()

tickers = ['SPY', 'GME', 'AMZN', 'QQQ', 'TSLA', 'NVDA']

opens = {}
dates = {}
for tick in tickers:
    tick_df = yf.Ticker(tick).history(
            period='2y',
            interval='1d',
            actions=False)
    dates[tick] = tick_df.index.values
    opens[tick] = tick_df['Open'].values

log_returns = {}
for tick, vals in opens.items():
    log_returns[tick] = np.diff(
            np.log(vals))

df_gme = pd.DataFrame(data=log_returns['GME'], columns=['GME'])
df_spy = pd.DataFrame(data=log_returns['SPY'], columns=['SPY'])
df_qqq = pd.DataFrame(data=log_returns['QQQ'], columns=['QQQ'])
df_nvda = pd.DataFrame(data=log_returns['NVDA'], columns=['NVDA'])

mu_spy = df_spy['SPY'].values.mean()
std_spy = df_spy['SPY'].values.std()
spy_rv = norm(loc=mu_spy, scale=std_spy)
t_spy = np.linspace(spy_rv.ppf(.001), spy_rv.ppf(.999), 1000)

mu_nvda = df_nvda['NVDA'].values.mean()
std_nvda = df_nvda['NVDA'].values.std()
nvda_rv = norm(loc=mu_nvda, scale=std_nvda)
t_nvda = np.linspace(nvda_rv.ppf(.001), nvda_rv.ppf(.999), 1000)

fig = make_subplots(rows=2, cols=2,
                    subplot_titles=[
                        'GME', 'SPY', 'QQQ', 'NVDA'])
_ = fig.add_trace(
        go.Histogram(x=df_gme['GME'],
                     nbinsx=70,
                     name='GME'),
        row=1, col=1)
_ = fig.add_trace(
        go.Histogram(x=df_spy['SPY'],
                     nbinsx=70,
                     name='SPY'),
        row=1, col=2)
_ = fig.add_trace(
        go.Scatter(x=t_spy,
                   y=spy_rv.pdf(t_spy)),
        row=1, col=2)
_ = fig.add_trace(
        go.Histogram(x=df_qqq['QQQ'],
                     nbinsx=70,
                     name='QQQ'),
        row=2, col=1)
_ = fig.add_trace(
        go.Histogram(x=df_nvda['NVDA'],
                     nbinsx=70,
                     name='NVDA'),
        row=2, col=2)
_ = fig.add_trace(
        go.Scatter(x=t_nvda,
                   y=nvda_rv.pdf(t_nvda)),
        row=2, col=2)
fig.update_layout(
        title={'text': 'Distribution of log returns',
               'x': .5})
fig.show()


fig = px.histogram(data_frame=df_spy, x=f'SPY', nbins=40)
fig.show()
