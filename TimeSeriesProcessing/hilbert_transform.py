import numpy as np
from scipy.signal import hilbert
import matplotlib.pyplot as plt

time = np.linspace(0, 1, 1000)
data = lambda x : np.sin(2 * np.pi * 3 * x) + np.cos(2 * np.pi * 9 * x)
signal = data(time)
signal_a = hilbert(signal)

freqs = np.fft.fftfreq(len(signal_a))
fft_a = np.fft.fft(signal_a)
fft = np.fft.fft(signal)


fig, axs = plt.subplots(2, 2)
plt.suptitle('Original and analytic signal')
axs[0][0].set_title('Original signal : x(t)')
axs[0][0].plot(time, signal)
axs[1][0].plot(time, np.abs(signal), label='|x(t)|')
axs[1][0].legend(loc='upper right')
axs[0][1].set_title('Analytic signal : x_a(t) = x(t) + jHTx(t)')
axs[0][1].plot(time, signal_a.real, label='x(t)')
axs[0][1].plot(time, signal_a.imag, label='HTx(t)')
axs[0][1].legend(loc='upper right')
axs[1][1].plot(time, np.abs(signal_a), label='|x_a(t)|')
axs[1][1].legend(loc='upper right')
plt.show()

fig, axs = plt.subplots(2, 2)
plt.suptitle('Original and analytic signal : x(t) and |Fx(f)|')
axs[0][0].set_title('Original signal : x(t)')
axs[0][0].plot(time, signal)
axs[1][0].plot(freqs, np.abs(fft))
axs[0][1].set_title('Analytic signal : z(t) = x(t) + jHTx(t)')
axs[0][1].plot(time, signal_a.real, label='x(t)')
axs[0][1].plot(time, signal_a.imag, label='HTx(t)')
axs[0][1].legend(loc='upper right')
axs[1][1].plot(freqs, np.abs(fft_a))
plt.show()



