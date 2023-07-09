import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

x = np.linspace(0, 1, 5000)

def bump(x):
    return np.exp(-(x ** 2))

def type1(x):
    val = bump(14 * (x - .7))
    val += bump(14 * (x - .2)) / 3
    return val

def type2(x):
    val = bump(14 * (x - .5))
    return val

def type3(x):
    val = bump(14 * (x - .2))
    val += bump(14 * (x - .7)) / 3
    return val

data, data_labels = [], []
for i in range(3000):
    x_ = x + np.random.normal(0, .01)
    x_ *= np.random.normal(1, .01)
    h = np.random.normal(1, .01, size=(3, 5000))
    data.append(
        type1(x_) + h[0]
    )
    data.append(
        type2(x_) + h[1]
    )
    data.append(
        type3(x_) + h[2]
    )
    data_labels += [1, 2, 3]

data = np.array(data)
data_labels = np.array(data_labels)

for d in data[:: 200]:
    plt.plot(d)
plt.show()

data_df = pd.DataFrame(
    data.T,
    columns=data_labels
)

data_df.to_csv('time_series.csv')
