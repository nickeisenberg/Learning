import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize

time = np.linspace(0, 10, 10000)

data = lambda x : np.sin(2 * np.pi * x) + 1
data2 = lambda x : 10 * np.exp((-(x - 4) ** 2) / .0011) + 10 * np.exp((-(x - 6) ** 2) / .0011)

data = data(time)
data2 = data2(time)
data += data2

def area(data):
    part = len(data) - 1
    areas = []
    for i in range(part):
        areas.append(data[i + 1] / len(data))
    return np.sum(np.array(areas))

# area = area(data)
# print(f'area under the curve : {area}')
# print(f'mean of data : {np.mean(data)}')

from copy import deepcopy
med_count = 0
data_ = deepcopy(data)
for i in range(200):
    data_ += np.roll(data, i + 1)
    data_ += np.roll(data, i - 1)
    med_count += 2

data_ /= med_count

fig, axs = plt.subplots(1, 3)
axs[0].plot(time, data)
axs[0].set_title('data')
axs[1].plot(time, data_)
axs[1].set_title('long term average')
axs[2].plot(time, data - data_)
axs[2].set_title('data - long term average')
plt.show()

