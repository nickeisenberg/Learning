import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt, wiener

stock_info = yf.Ticker('GME')

price_history = stock_info.history(period='2y',
                                 interval='1h',
                                 actions=False)

open_price = price_history['Open'].values

# Min/Max scaler
max_open = np.max(open_price)
min_open = np.min(open_price)
open_price -= min_open
open_price /= max_open - min_open

# Wiener filtered data
open_price_wiener = wiener(open_price, 9)

# median filtered data
open_price_med = medfilt(open_price, 9)

# Plotting a spectrogram
time_window = 256
time_step = 256
time_unit = 1

spectrogram = []
spec_times = []
freqs = np.fft.rfftfreq(time_window, d=time_unit)

for time_start in np.arange(0, len(open_price), time_step):

    data_time_ref = time_start + time_step / 2
    data = open_price[time_start : time_start + time_step]

    if len(data) < time_window:
        continue

    spec_times.append(data_time_ref)
    spectrogram.append(np.abs(np.fft.rfft(data)))

spectrogram = np.array(spectrogram).T
spec_times = np.array(spec_times)

# fig, ax = plt.subplots()
# x2d, y2d = np.meshgrid(spec_times, freqs)
# pc = ax.pcolormesh(spec_times, freqs, 10 * np.log10(spectrogram + .001), shading='auto')
# fig.colorbar(pc)
# plt.show()


'''
time = np.linspace(0, 2, len(open_price))

print(len(open_price))
freq = np.fft.rfftfreq(len(open_price))
fft = abs(np.fft.rfft(open_price))

plt.subplot(121)
plt.plot(time, open_price)

plt.subplot(122)
plt.plot(freq[1:], fft[1:])

plt.show()

'''
