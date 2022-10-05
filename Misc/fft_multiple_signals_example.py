'''
A program that illustrates what fft does. The fft can take noise data versus the time domian and
then transform it to the frequency domain. The plots  versus the frequency domain will have peaks
roughly corresponding to the number of signals from the data and each peak will be roughly centered
around the period of the frequency.
'''

import numpy as np
import matplotlib.pyplot as plt

part_size = 500
t = np.linspace(0, 10, part_size)
noise = np.random.randn(part_size)

# Cosine data
data_cos = np.cos(5 * np.pi * t) + noise
fft = np.fft.rfft(data_cos)
fft_abs = abs(np.fft.rfft(data_cos))
freq = np.fft.rfftfreq(part_size)

plt.subplot(221)
plt.plot(t, data_cos)
plt.title('data')

plt.subplot(222)
plt.plot(freq, fft_abs)
plt.title('abs(fft)')

plt.subplot(223)
plt.plot(freq, fft.real)
plt.title('fft.real')

plt.subplot(224)
plt.plot(freq, fft.imag)
plt.title('fft.imag')

plt.suptitle(f'data : x(t) = cos(5 $\pi$ t)\n0 < t < 10 with a partition size of {part_size}')
plt.show()

# Sine data
data_sin = np.sin(2 * np.pi * t) + noise
fft = np.fft.rfft(data_sin)
fft_abs = abs(np.fft.rfft(data_sin))
freq = np.fft.rfftfreq(part_size)

plt.subplot(221)
plt.plot(t, data_sin)
plt.title('data')

plt.subplot(222)
plt.plot(freq, fft_abs)
plt.title('abs(fft)')

plt.subplot(223)
plt.plot(freq, fft.real)
plt.title('fft.real')

plt.subplot(224)
plt.plot(freq, fft.imag)
plt.title('fft.imag')

plt.suptitle(f'data : x(t) = sin(2 $\pi$ t)\n0 < t < 10 with a partition size of {part_size}')
plt.show()

# Sine + Cosine data
data_plus = data_cos + data_sin - noise
fft = np.fft.rfft(data_plus)
fft_abs = abs(np.fft.rfft(data_plus))
freq = np.fft.rfftfreq(part_size)

plt.subplot(221)
plt.plot(t, data_plus)
plt.title('data')

plt.subplot(222)
plt.plot(freq, fft_abs)
plt.title('abs(fft)')

plt.subplot(223)
plt.plot(freq, fft.real)
plt.title('fft.real')

plt.subplot(224)
plt.plot(freq, fft.imag)
plt.title('fft.imag')

plt.suptitle(f'data : x(t) = cos(5 $\pi$ t) + sin(2 $\pi$ t)\n0 < t < 10 with a partition size of {part_size}')
plt.show()
# sample_sum = np.zeros(part_size)
# for i in range(1,4):
#     sample = np.sin(10 * i * np.pi * t)
#     sample_sum += sample
#     data = sample + noise
#     fft = abs(np.fft.rfft(data))
#     freq = np.fft.rfftfreq(part_size)
#     plt.subplot(1, 4, i)
#     plt.plot(freq, fft)
# 
# data_sum = sample_sum + noise
# plt.subplot(144)
# plt.plot(freq, abs(np.fft.rfft(data_sum)))

