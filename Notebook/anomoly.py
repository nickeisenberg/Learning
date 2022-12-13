import numpy as np
import matplotlib.pyplot as plt

np.random.seed(1)

time = np.linspace(0, 3, 1000)

def bump(x, s):
    b = np.exp(- (x - s) ** 2 / .005) / np.sqrt(.005)
    return b

data = np.zeros(1000)
for i in np.arange(0, 3, 1 / 3)[1:]:
    data += bump(time, i)
data /= max(data)
anomoly = bump(time, 2)
anomoly /= 25
data += anomoly

