import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, curve_fit

# create some known signal to curve fit
time = np.linspace(0, 1, 501)
data = np.cos(2 * np.pi * 4 * time) + np.cos(2 * np.pi * 9 * time) + np.cos(2 * np.pi * 20 * time)
noise = np.sqrt(1 / 15) * np.random.randn(501)
signal = data + noise

# create the approximating function
# def cos_sum(x, A, B, C):
#     if not all([isinstance(i, (np.ndarray, list)) for i in [A, B, C]]):
#         return print('A, B, C are not lists or arrays')
# 
#     if (len(A) + len(B) + len(C)) / 3 != float(len(A)):
#         return print('lengths of A, B, C are not the same')
# 
#     sums = []
#     for a, b, c in zip(A, B, C):
#         sums.append(a * np.cos(b * (x - c)))
# 
#     sums = np.array(sums)
# 
#     return np.sum(sums, axis=0)

def cos_sum(x, P):
    assert isinstance(P, np.ndarray)

    sums = []
    for i in range(0, P.shape[0], 3):
        a, b, c = P[i], P[i+1], P[i+2]
        sums.append(a * np.cos(b * (x - c)))

    sums = np.array(sums)

    return np.sum(sums, axis=0)

# create a residual function for minimize
# def resid(params, x):
#     fit = cos_sum(x, *params)
#     residual = np.sqrt(np.mean(np.abs(fit - signal)) ** 2)
#     return residual

def resid(params, x):
    assert isinstance(params, np.ndarray)
    fit = cos_sum(x, params)
    residual = np.sqrt(np.mean(np.abs(fit - signal)) ** 2)
    return residual

# create a guess for minimize 
# guess_A = np.random.normal(1, .2, size=3)
# guess_B = 2 * np.pi * np.array([4, 9, 20], dtype=float)
# guess_C = np.random.normal(0, .2, size=3)
# guess = [guess_A, guess_B, guess_C]

guess_A = np.random.normal(1, .2, size=3)
guess_B = 2 * np.pi * np.array([3.8, 9.5, 18], dtype=float)
guess_C = np.random.normal(0, .2, size=3)
guess = np.vstack([guess_A, guess_B, guess_C]).T
guess = np.reshape(guess, (1, guess.shape[0] * guess.shape[1]))

optimization = minimize(resid, guess, args=(time))
params = optimization.x
err = np.max(np.abs(signal - cos_sum(time, params)))
plt.subplot(131)
plt.plot(time, signal)
plt.plot(time, cos_sum(time, params), label=f'None: {err}')
plt.legend(loc='upper center')

optimization = minimize(resid, guess, args=(time), method='Nelder-Mead')
params = optimization.x
err = np.max(np.abs(signal - cos_sum(time, params)))
plt.subplot(132)
plt.plot(time, signal)
plt.plot(time, cos_sum(time, params), label=f'Nelder-Mead: {err}')
plt.legend(loc='upper center')

optimization = minimize(resid, guess, args=(time), method='Powell')
params = optimization.x
err = np.max(np.abs(signal - cos_sum(time, params)))
plt.subplot(133)
plt.plot(time, signal)
plt.plot(time, cos_sum(time, params), label=f'Powell: {err}')
plt.legend(loc='upper center')

plt.show()
