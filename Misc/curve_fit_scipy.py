import numpy as np
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt

# create a signal to curve fit
time = np.linspace(0, 1, 61)
noise = np.random.randn(61)
data = np.sin(2 * np.pi * 3 * time)
signal = data + noise

# create some functions to try for the curve fit
def cos(x, a, b, c):
    return a * np.cos(b * (x - c))

# def cos_sum(x, a, b, c):
#     if isinstance(a
#     print(a, b, c)
#     return None


a = np.array([1, 2])
b = np.array([3, 4])

print( all(isinstance(i, np.ndarray) for i in [a, b]) )


'''
plt.plot(time, signal, label='signal')
plt.plot(time, data, label='true data')
plt.legend(loc='upper right')
plt.show()
'''

