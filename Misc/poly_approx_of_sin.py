import numpy as np
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def approx_3(coef, n):
    partition = np.linspace(0, 2 * np.pi, n)
    a0, a1, a2, a3 = coef
    summands = []
    for t in partition:
        sum_i = abs((a0 + a1 * t + a2 * t ** 2 + a3 * t ** 3) - np.sin(t))
        summands.append(sum_i)
    summands = np.array(summands)
    return np.sum(summands)

def approx_k(coef, n):
    partition = np.linspace(0, 2 * np.pi, n)
    summands = []
    for t in partition:
        poly_k = []
        for a in range(len(coef)):
            poly_k.append(coef[a] * t ** a)
        poly_k = np.sum(np.array(poly_k))
        summands.append(abs(poly_k - np.sin(t)))
    summands = np.array(summands)
    return np.sum(summands)

guess = np.random.randn(4)
params = minimize(approx_3, guess, args=(100)).x

def poly(t, params):
    summands = []
    for a in range(len(params)):
        summands.append(params[a] * t ** a)
    summands = np.array(summands)
    return np.sum(summands)

##################################################

order = 5
guess = np.random.randn(order + 1)
sin_approx = minimize(approx_k, guess, args=(100))

params = sin_approx.x
time = np.linspace(0, 2 * np.pi, 300)

poly_ = []
for t in time:
    poly_.append(poly(t, params))

poly_ = np.array(poly_)


plt.plot(time, poly_, label='approximation')
plt.plot(time, np.sin(time), ':', c='red', label='sin(x)')
plt.title(f'An order {order} polynomial approximation of sin(x)')
plt.legend(loc='upper right')
plt.show()
