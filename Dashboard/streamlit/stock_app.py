import time
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import streamlit as st
import os

st.set_page_config(
        page_title='Faang',
        layout='wide',
        )

@st.cache_data
def get_data(path, OHLCV) -> pd.DataFrame:
    dfs = []
    for fn in os.listdir(path):
        if fn.endswith('.csv') and OHLCV in fn:
            dfs.append(pd.read_csv(
                os.path.join(path, fn), index_col=0))
    return pd.concat(dfs).sort_index()

path = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/faang/filt/'
path_sp = '/Users/nickeisenberg/GitRepos/Phython/yfDataFrames/sp_500/filt/'

df = get_data(path, 'Open')
df_sp = get_data(path_sp, 'Open')

st.title('Faang stock data')

ticker_filter = st.selectbox('Select the ticker',
                             df.columns.values)

placeholder = st.empty()

stock_price = df[ticker_filter].values
dates = df.index.values
inds = np.arange(0, df.shape[0], 1)

fig = go.Figure()
fig_sp = go.Figure()
for ind0, ind1 in zip(inds[::60][:-1], inds[::60][1:]):

    price = stock_price[ind0: ind1]
    price_inds = inds[ind0: ind1]

    price_sp = df_sp['SPY_Open'].values[ind0: ind1]

    with placeholder.container():

        fig_col0, fig_col1 = st.columns(2)
        with fig_col0:
            _ = fig.add_trace(go.Scatter(x=price_inds, y=price,
                                         showlegend=False,
                                         line={'color': 'red'}))
            _ = fig.update_layout(
                    title={'text': f'{ticker_filter}',
                           'x': .5}
                    )
            st.write(fig)

        with fig_col1:
            _ = fig_sp.add_trace(go.Scatter(x=price_inds, y=price_sp,
                                            showlegend=False,
                                            line={'color': 'blue'}))
            _ = fig_sp.update_layout(
                    title={'text': 'SPY',
                           'x': .5}
                    )
            st.write(fig_sp)





