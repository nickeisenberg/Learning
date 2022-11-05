import numpy as np
from scipy.optimize import minimize, curve_fit
import matplotlib.pyplot as plt

np.random.seed(1)
# create a signal to curve fit
time = np.linspace(0, 1, 101)
noise = np.random.randn(101)

# cos data
# data = np.cos(2 * np.pi * 3 * time)
# signal = data + noise

# cos + sin data
data = np.cos(2 * np.pi * 3 * time) + np.sin(2 * np.pi * 2 * time)
signal = data + noise

# create some functions to try for the curve fit
def cos(x, a, b, c):
    return a * np.cos(b * (x - c))

def cos_plus_sin(x, Ac, Bc, Cc, As, Bs, Cs):
#     if not all([isinstance(i, (np.ndarray, list)) for i in [Ac, Bc, Cc, As, Bs, Cs]]):
#         return print('Ac, Bc, Cc, As, Bs, Cs are not all interable')
# 
#     if not all([len(i) == len(Ac) for i in [Ac, Bc, Cc, As, Bs, Cs]]):
#         return print('lengths of Ac, Bc, Cc, As, Bs, Cs must match')

#     if (len(A_c) + len(B_) + len(C)) / 3 != float(len(A)):
#         return print('lengths of Ac, Bc, Cc, As, Bs, Cs must match')

#     sums_cos = []
#     for a, b, c in zip(Ac, Bc, Cc):
#         sums_cos.append(a * np.cos(b * (x - c)))
#     sums_cos = np.array(sums_cos)
#     sums_cos = np.sum(sums_cos, axis=0)
# 
#     sums_sin = []
#     for a, b, c in zip(As, Bs, Cs):
#         sums_sin.append(a * np.sin(b * (x - c)))
#     sums_sin = np.array(sums_sin)
#     sums_sin = np.sum(sums_sin, axis=0)

    return Ac * np.cos(Bc * (x - Cc)) + As * np.sin(Bs * (x - Cs)) 

'''
# Testing the cos_plus_sin function
#-------------------------
x = np.linspace(0, 1, 101)
data = cos_plus_sin(x, [1], [2*np.pi*3], [0], [1], [2*np.pi*11], [0])
freq = np.fft.rfftfreq(len(x), d=1/100)
fft = abs(np.fft.rfft(data))
plt.subplot(121)
plt.plot(x, data)
plt.subplot(122)
plt.plot(freq, fft)
plt.show()
#-------------------------
'''

def cos_sum(x, A, B, C):
    if not all([isinstance(i, (np.ndarray, list)) for i in [A, B, C]]):
        return print('A, B, C are not all interable')

    if (len(A) + len(B) + len(C)) / 3 != float(len(A)):
        return print('lengths of A, B, C must match')

    sums = []
    for a, b, c in zip(A, B, C):
        sums.append(a * np.cos(b * (x - c)))

    sums = np.array(sums)

    return np.sum(sums, axis=0)

'''
# testing the cos_sum function
#-------------------------
a = np.array([1, 1])
b = np.array([2 * np.pi, 2 * np.pi * 4])
c = [0, 0]

x = np.linspace(0, 1, 101)
data = cos_sum(x, a, b ,c)
freq = np.fft.rfftfreq(len(x), 1/100)
fft = abs(np.fft.rfft(data))

plt.subplot(121)
plt.plot(x, data)
plt.subplot(122)
plt.plot(freq, fft)
plt.show()
#-------------------------
'''

# prep residual functions for minimize method
# edit res depending on the clean data
def resid_sup(C, x):
    res = np.max(np.abs(signal - cos_plus_sin(x, *C)))
    return res

def resid_L1(C, x):
    res = np.mean(np.abs(signal - cos_plus_sin(x, *C)))
    return res

def resid_L2(C, x):
    res = np.sqrt(np.mean(np.abs(signal - cos_plus_sin(x, *C)) ** 2))
    return res

#--------------------Results--------------------

# The guess is far more forgiving for the L1 and L2 norms than the sup norm.

# Set up for the cos_plus_sin data

params_sup = minimize(resid_sup, [1, 18, 0, 1, 12, 0], args=(time))
params_L1 = minimize(resid_L1, [1, 18, 0, 1, 12, 0], args=(time))
params_L2 = minimize(resid_L2, [1, 18, 0, 1, 12, 0], args=(time))
# for i in range(10):
#     params_sup = minimize(resid_sup, params_sup.x, args=(time))
#     params_L1 = minimize(resid_L1, params_L1.x, args=(time))
#     params_L2 = minimize(resid_L2, params_L2.x, args=(time))

# Error
true_error = np.max(np.abs(data - signal))
fit_error_sup =  np.max(np.abs(cos_plus_sin(time, *params_sup.x)))
fit_error_L1 =  np.max(np.abs(cos_plus_sin(time, *params_L1.x)))
fit_error_L2 =  np.max(np.abs(cos_plus_sin(time, *params_L2.x)))

plt.subplot(222)
plt.title('minimize method with sup norm')
plt.plot(time, data, label='clean signal')
plt.plot(time, signal, label='signal with noise')
plt.plot(time, cos_plus_sin(time, *params_sup.x), label='fit', linestyle='--')
plt.plot([], [], label=f'fit error: {fit_error_sup}')
plt.legend(loc='upper right')

plt.subplot(223)
plt.title('minimize method with L1 norm')
plt.plot(time, data, label='clean signal')
plt.plot(time, signal, label='signal with noise')
plt.plot(time, cos_plus_sin(time, *params_L1.x), label='fit', linestyle='--')
plt.plot([], [], label=f'fit error: {fit_error_L1}')
plt.legend(loc='upper right')

plt.subplot(224)
plt.title('minimize method with L2 norm')
plt.plot(time, data, label='clean signal')
plt.plot(time, signal, label='signal with noise')
plt.plot(time, cos_plus_sin(time, *params_L2.x), label='fit', linestyle='--')
plt.plot([], [], label=f'fit error: {fit_error_L2}')
plt.legend(loc='upper right')

# prep for curve_fit method
popt, pcov = curve_fit(cos_plus_sin,
                       time, signal,
                       bounds=([0, 15, -.5, 0, 10, -.5], [1.5, 20, .5, 1.5, 13, .5]))
fit_error_cf =  np.max(np.abs(cos_plus_sin(time, *popt)))

plt.subplot(221)
plt.title('curve_fit method')
plt.plot(time, data, label='clean signal')
plt.plot(time, signal, label='signal with noise')
plt.plot(time, cos_plus_sin(time, *popt), label='fit', linestyle='--')
plt.plot([], [], label=f'fit error: {fit_error_cf}')
plt.legend(loc='upper right')

plt.suptitle(f'$ \cos(2 \pi \cdot 3 t) + \sin(2 \pi \cdot 2 t) + noise$\n \
             true error: {true_error}')
plt.show()


'''
# Set up for the cos data

params_sup = minimize(resid_sup, [1, 18, 0], args=(time))
params_L1 = minimize(resid_L1, [1, 18, 0], args=(time))
params_L2 = minimize(resid_L2, [1, 18, 0], args=(time))

plt.subplot(222)
plt.title('minimize method with sup norm')
plt.plot(time, data, label='clean signal')
plt.plot(time, signal, label='signal with noise')
plt.plot(time, cos(time, *params_sup.x), label='fit', linestyle='--')
plt.legend(loc='upper right')

plt.subplot(223)
plt.title('minimize method with L1 norm')
plt.plot(time, data, label='clean signal')
plt.plot(time, signal, label='signal with noise')
plt.plot(time, cos(time, *params_L1.x), label='fit', linestyle='--')
plt.legend(loc='upper right')

plt.subplot(224)
plt.title('minimize method with L2 norm')
plt.plot(time, data, label='clean signal')
plt.plot(time, signal, label='signal with noise')
plt.plot(time, cos(time, *params_L2.x), label='fit', linestyle='--')
plt.legend(loc='upper right')

# prep for curve_fit method
popt, pcov = curve_fit(cos, time, signal, bounds=([0, 15, -.5], [1.5, 20, .5]))
plt.subplot(221)
plt.title('curve_fit method')
plt.plot(time, data, label='clean signal')
plt.plot(time, signal, label='signal with noise')
plt.plot(time, cos(time, *popt), label='fit', linestyle='--')
plt.legend(loc='upper right')

plt.show()
'''
