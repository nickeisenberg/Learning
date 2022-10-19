import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np
from scipy.signal import medfilt, wiener, find_peaks, peak_widths, butter, sosfiltfilt
from sklearn.preprocessing import MinMaxScaler 

stock_info = yf.Ticker('GME')

price_history = stock_info.history(period='2y',
                                 interval='1d',
                                 actions=False)

price_history['Open'].to_csv('/Users/nickeisenberg/GitRepos/Python_Misc/Misc/Plots/gme_open.csv')
open_price = price_history['Open'].values

# Min/Max scaler
max_open = np.max(open_price)
min_open = np.min(open_price)
open_price -= min_open
open_price /= (max_open - min_open)

# Wiener filtered data
open_price_wiener = wiener(open_price, 9)

# median filtered data
open_price_med = medfilt(open_price, 9)

# #-Plotting-a-spectrogram---------------------------
# time_window = 256
# time_step = 256
# time_unit = 1
# 
# spectrogram = []
# spec_times = []
# freqs = np.fft.rfftfreq(time_window, d=time_unit)
# 
# for time_start in np.arange(0, len(open_price), time_step):
# 
#     data_time_ref = time_start + time_step / 2
#     data = open_price[time_start : time_start + time_step]
# 
#     if len(data) < time_window:
#         continue
# 
#     spec_times.append(data_time_ref)
#     spectrogram.append(np.abs(np.fft.rfft(data)))
# 
# spectrogram = np.array(spectrogram).T
# spec_times = np.array(spec_times)
# 
# fig, ax = plt.subplots()
# x2d, y2d = np.meshgrid(spec_times, freqs)
# pc = ax.pcolormesh(spec_times, freqs, 10 * np.log10(spectrogram + .001), shading='auto')
# fig.colorbar(pc)
# plt.show()
# #--------------------------------------------------

# #-Combined-frequency-profile-----------------------
# time_window = 64
# time_step = 64
# time_unit = 1
# 
# spectrogram = []
# spec_times = []
# freqs = np.fft.rfftfreq(time_window, d=time_unit)
# 
# for time_start in np.arange(0, len(open_price), time_step):
# 
#     data_time_ref = time_start + time_step / 2
#     data = open_price[time_start : time_start + time_step]
# 
#     if len(data) < time_window:
#         continue
# 
#     spec_times.append(data_time_ref)
#     spectrogram.append(np.abs(np.fft.rfft(data)))
# 
# spectrogram = np.array(spectrogram)
# spectrogram = np.percentile(spectrogram, 90,  axis=0)
# spec_times = np.array(spec_times)
# 
# plt.subplot(133)
# plt.plot(freqs, spectrogram)
# #--------------------------------------------------

#-Finding-peaks-and-bandwidths-for-butter-filters--
time = np.linspace(0, 2, len(open_price))
freq = np.fft.rfftfreq(len(open_price), d = time[1] - time[0])
fft = abs(np.fft.rfft(open_price))
peak_idx, peak_dict = find_peaks(fft)
peak_pairs = []
peak_val = []
peak_dom = []
for pi in peak_idx:
    peak_dom.append(freq[pi])
    peak_val.append(fft[pi])
    peak_pairs.append([freq[pi], fft[pi]])

plt.plot(freq, fft)
plt.scatter(peak_dom, peak_val, marker='x', c='red')
plt.show()

print(peak_pairs[:15])
'''
There seem to be distinct frequencies at around...
... 2.495, 4.49, 6.48, 8.48, 16.966. We can first use a lowpass filter...
... and try to isolate them.
'''
# lowpass
# sos_l = butter(9, 25, 'lowpass', fs=len(open_price), output='sos')
# filt_data_l = sosfiltfilt(sos_l, open_price)

'''
The lowpass with cutoff 20 removed the signal with freq=17. Not sure why.
We can use bandpass filters to look at each of the above mentioned frequencies.
'''
sos_bp = butter(9, [16, 18], 'bandpass', fs=len(open_price), output='sos')
filt_data_bp = sosfiltfilt(sos_bp, open_price)
max_bp = np.max(filt_data_bp)
min_bp = np.min(filt_data_bp)
filt_data_bp -= min_bp
filt_data_bp /= (max_bp - min_bp)
bp_implied_noise = open_price - filt_data_bp
#--------------------------------------------------

plt.subplot(231)
plt.plot(time, open_price)

plt.subplot(232)
plt.plot(time, filt_data_bp)

plt.subplot(233)
fft_filt = abs(np.fft.rfft(filt_data_bp))
plt.plot(freq, fft_filt)

plt.subplot(234)
plt.plot(time, bp_implied_noise)

plt.show()
