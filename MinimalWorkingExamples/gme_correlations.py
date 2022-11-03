from sys import exit
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import scipy.signal as sps
from sklearn.preprocessing import MinMaxScaler

gme_2y = pd.read_csv('/Users/nickeisenberg/GitRepos/Python_Misc/MinimalWorkingExamples/DataSets/gme_11_3_22.csv')
gme_2y = gme_2y['Open'].values.reshape((-1,1))
Mm = MinMaxScaler(feature_range=(0,1))
gme_scaled = Mm.fit_transform(gme_2y)

time = np.linspace(0, 1, len(gme_2y)).reshape((-1,1))

low = int(.9182 * len(time))
up = int(.9835 * len(time))

gme_pat = gme_scaled[low:up]
gme_pat_norm = gme_pat / np.sqrt(np.sum(np.multiply(gme_pat, gme_pat)))
pat_len = len(gme_pat_norm)

corr_scores = []
for i in range(0, low + 1):
    gme_ref = gme_scaled[i : i + pat_len]
    gme_ref_norm = gme_ref / np.sqrt(np.sum(np.multiply(gme_ref, gme_ref)))
    score = np.sqrt(np.sum(np.multiply(gme_ref_norm, gme_pat_norm)))
    corr_scores.append([i, score])
corr_scores = np.array(corr_scores)
corr_scores = corr_scores[corr_scores[:, 1].argsort()][::-1]

top_scores = corr_scores[::int(pat_len/2)][:10]
ind = int(top_scores[:, 0][1])

corr_past = gme_2y[ind : ind + pat_len]
time_past = time[ind : ind + pat_len]
corr_fut = gme_2y[ind + pat_len : ind + 2 * pat_len]
time_fut = time[ind + pat_len : ind + 2 * pat_len]

plt.subplot(121)
plt.plot(time_past, corr_past)
plt.plot(time_fut, corr_fut)

plt.subplot(122)
plt.plot(time, gme_2y)
plt.show()
