import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit
import pandas as pd

# create some known signal to curve fit
time = np.linspace(0, 1, 501)
data = np.cos(2 * np.pi * 4 * time) + np.cos(2 * np.pi * 9 * time) + np.cos(2 * np.pi * 20 * time)
noise = np.sqrt(1 / 15) * np.random.randn(501)
signal = data + noise

def cos_sum(x, P):
    assert isinstance(P, np.ndarray)

    sums = []
    for i in range(0, P.shape[0], 3):
        a, b, c = P[i], P[i+1], P[i+2]
        sums.append(a * np.cos(b * (x - c)))

    sums = np.array(sums)

    return np.sum(sums, axis=0)

def resid(params, x):
    assert isinstance(params, np.ndarray)
    fit = cos_sum(x, params)
    residual = np.sqrt(np.mean(np.abs(fit - signal)) ** 2)
    return residual

guess_A = np.random.normal(1, .2, size=3)
guess_B = 2 * np.pi * np.array([3.8, 9.5, 18], dtype=float)
guess_C = np.random.normal(0, .2, size=3)
guess = np.vstack([guess_A, guess_B, guess_C]).T
guess = np.reshape(guess, (1, guess.shape[0] * guess.shape[1]))

err_coef = []
index = []

optimization = minimize(resid, guess, args=(time))
params = optimization.x
index.append('None')
err = np.max(np.abs(signal - cos_sum(time, params)))
err_coef.append(np.hstack((err, params)))
plt.subplot(131)
plt.plot(time, signal)
plt.plot(time, cos_sum(time, params), label=f'None: {err}')
plt.legend(loc='upper center')

optimization = minimize(resid, guess, args=(time), method='Nelder-Mead')
params = optimization.x
index.append('Nelder-Mead')
err = np.max(np.abs(signal - cos_sum(time, params)))
err_coef.append(np.hstack((err, params)))
plt.subplot(132)
plt.plot(time, signal)
plt.plot(time, cos_sum(time, params), label=f'Nelder-Mead: {err}')
plt.legend(loc='upper center')

optimization = minimize(resid, guess, args=(time), method='Powell')
params = optimization.x
index.append('Powell')
err = np.max(np.abs(signal - cos_sum(time, params)))
err_coef.append(np.hstack((err, params)))
plt.subplot(133)
plt.plot(time, signal)
plt.plot(time, cos_sum(time, params), label=f'Powell: {err}')
plt.legend(loc='upper center')


df = pd.DataFrame(np.array(err_coef),
                  columns=['sup-norm error',
                           'a1', 'b1', 'c1',
                           'a2', 'b2', 'c2',
                           'a3', 'b3', 'c3'],
                  index=index)
df.to_csv('/Users/nickeisenberg/GitRepos/Python_Misc/Misc/Plots/coef.csv')


plt.show()
