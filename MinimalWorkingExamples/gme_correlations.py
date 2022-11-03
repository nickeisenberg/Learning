from sys import exit
import pandas as pd
import numpy as np
np.set_printoptions(suppress=True)
import matplotlib.pyplot as plt
import scipy.signal as sps
from sklearn.preprocessing import MinMaxScaler

gme_2y = pd.read_csv('/Users/nickeisenberg/GitRepos/Python_Misc/MinimalWorkingExamples/DataSets/gme_11_3_22.csv')
cols = gme_2y.columns
gme_2y.rename(columns={cols[0] : 'Date'}, inplace=True)
dates = gme_2y['Date'].values

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

top_scores = [] # index for the start of the correlation reference
for i in corr_scores[:, 0]:
    if np.abs(i -low) < pat_len:
        continue

    if len(top_scores) == 0:
        top_scores.append(i)
        continue

    dists_i = []
    for ts in top_scores:
        dists_i.append(np.abs(ts - i))
    dists_i = np.array(dists_i)
    if np.min(dists_i) >= pat_len / 2: # chose how close each start point can be
        top_scores.append(i)

    if len(top_scores) == 8:
        break

for i in range(8):
    ind = int(top_scores[i])
    corr_past = gme_2y[ind : ind + pat_len]
    time_past = time[ind : ind + pat_len]
    corr_fut = gme_2y[ind + pat_len : ind + 3 * pat_len]
    time_fut = time[ind + pat_len : ind + 3 * pat_len]
    plt.subplot(3, 3, i+1)
    plt.plot(time_past, corr_past, c='blue')
    plt.plot(time_fut, corr_fut, c='darkorange')

plt.subplot(339)
plt.plot(time[low:up], gme_2y[low:up], c='green')
plt.plot(time[up:], gme_2y[up:], c='darkorange')

# for i in range(5):
#     ind = int(top_scores[i])
#     corr_past = gme_2y[ind : ind + pat_len]
#     time_past = time[ind : ind + pat_len]
#     corr_fut = gme_2y[ind + pat_len : ind + 3 * pat_len]
#     time_fut = time[ind + pat_len : ind + 3 * pat_len]
#     plt.subplot(2, 3, i+1)
#     plt.title(f'Start : {dates[ind]}\nEnd : {dates[ind + 3 * pat_len]}\n Blue to Green Corr : {corr_scores[ind][1]}')
#     plt.plot(time_past, corr_past, c='blue')
#     plt.plot(time_fut, corr_fut, c='darkorange')
# 
# plt.subplot(236)
# plt.title(f'Start : {dates[low]}\nEnd : {dates[-1]}\nGreen : Pattern for correlation')
# plt.plot(time[low:up], gme_2y[low:up], c='green')
# plt.plot(time[up:], gme_2y[up:], c='darkorange')

plt.tight_layout()
plt.suptitle('Movement of GME following correlated price movements\n\
             Blue plots are highly correlated with the green plot\n\
             Orange plots are the subsequent price movements')
plt.show()
