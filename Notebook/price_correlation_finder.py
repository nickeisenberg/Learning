import numpy as np
np.set_printoptions(suppress=True)
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import yfinance as yf

def pearson_corr(x, y):
    ux, uy = np.mean(x), np.mean(y)
    x_cen, y_cen = x - ux, y - uy
    return np.dot(x_cen, y_cen) / (np.sqrt(np.sum(np.square(x_cen))) * np.sqrt(np.sum(    np.square(y_cen))))

def norm_dot(x, y):
    return np.dot(x, y) / (np.sqrt(np.sum(np.square(x))) * np.sqrt(np.sum(np.square(y))))

class Correlation:

    def __init__(self, ticker=None, data_path=None):
        self.ticker = ticker
        self.data_path = data_path

    def find_corr(self,
                  period=None,
                  interval=None,
                  pat_start=None,
                  pat_end=None,
                  OHLC='Open',
                  prepost=False,
                  standardize=True):

        data_path = self.data_path
        ticker = self.ticker

        if isinstance(data_path, str):
            data = pd.read_csv(data_path)

        else:
            data = yf.Ticker(ticker).history(period=period,
                                             interval=interval,
                                             prepost=prepost,
                                             actions=False)

        data = data[OHLC].values.reshape((-1,1))

        if standardize:
            Mm = MinMaxScaler(feature_range=(0,1))
            data_scaled = Mm.fit_transform(data)

        if isinstance(pat_start, int) and isinstance(pat_end, int):
            pat = data[pat_start : pat_end]
            pat_s = data_scaled[pat_start : pat_end]
            pat_len = len(pat)
            pat_ind = [pat_start, pat_end]

        corr_scores = []
        for i in range(0, pat_ind[0] + 1):
            ref_s = data_scaled[i : i + pat_len]
            p_score_s = pearson_corr(pat_s.reshape(-1), ref_s.reshape(-1))
            corr_scores.append([i, p_score_s])
        corr_scores = np.array(corr_scores)
        corr_scores = corr_scores[corr_scores[:, 1].argsort()][::-1]

        top_scores = np.empty(0)
        for i in range(len(corr_scores[:, 0])):

            ref_ind = corr_scores[i][0]
            if np.abs(ref_ind - pat_ind[0]) < pat_len:
                continue

            if len(top_scores) == 0:
                top_scores = np.append(top_scores, [ref_ind, corr_scores[i][1]])
                top_scores = top_scores.reshape((1,2))
                continue

            dist_i = np.min(np.abs(top_scores[:, 0] - ref_ind))
            if dist_i >= pat_len / 5:
                top_scores = np.vstack((top_scores, [ref_ind, corr_scores[i][1]]))

            if len(top_scores) == 9:
                break
        top_scores = np.array(top_scores)
        self.top_scores = top_scores

        return top_scores

if __name__ == '__main__':

    data_path='/Users/nickeisenberg/GitRepos/Python_Misc/Notebook/DataSets/gme_11_3_22.csv'
    gme = Correlation(data_path=data_path)

    top_scores = gme.find_corr(pat_start=3232,
                               pat_end=3462)

    print(top_scores)
    print(gme.top_scores)








