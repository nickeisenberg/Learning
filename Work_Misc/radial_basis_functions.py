import numpy as np
import matplotlib.pyplot as plt

def rbf(x, a, k):
    cond0 = x < -1 / (2 * k)
    cond1 = (x >= -1 / (2 * k)) & (x <= 1 / (2 * k))
    cond2 = x > 1 / (2 * k)
    cond_list = [cond0, cond1, cond2]
    func = lambda x: .5 * (1 + np.sin(k * np.pi * x))
    func_list = [0, func, 1]
    return a * np.piecewise(x, cond_list, func_list)

time = np.linspace(-1/2, 1/2, 1000)
for k in [1, 2, 3]:
    plt.plot(time, rbf(time, 1, k))
plt.show()
