import numpy as np
import scipy.signal as sps
import matplotlib.pyplot as plt

def cos_signal_gen(dim=1, dom=[0,1], part_size=1000, freq=[1], noise=True):
    time = np.linspace(dom[0], dom[1], part_size + 1)
    dW = []
    if noise:
        for i in range(dim):
            dW_i = np.random.randn(len(time))
            dW.append(dW_i)
        dW = np.vstack(dW)

    desired = []
    for i in range(dim):
        des_i = []
        for i in range(len(freq)):
            des = np.cos(2 * np.pi * freq[i] * time)
            des_i.append(des)
        des_i = np.sum(np.array(des_i), axis=0)
        desired.append(des_i)
    desired - np.array(desired)

    signal = dW + desired

    return signal, desired, time, dW

signal, desired, time, _ = cos_signal_gen(dim=3, freq=[2, 9])





# plt.subplot(221)
# plt.plot(time,signal[0])
# 
# plt.subplot(223)
# plt.plot(time, desired[0])
# 
# plt.subplot(222)
# plt.plot(time,signal[1])
# 
# plt.subplot(224)
# plt.plot(time, desired[1])
# 
# plt.show()

