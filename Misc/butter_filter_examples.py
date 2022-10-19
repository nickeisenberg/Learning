import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.signal

time = np.linspace(0, 1, 1001)

sigs = []
sig_periods = [3, 10, 20]
data = np.zeros(len(time))

for i in range(len(sig_periods)):
    p = sig_periods[i]
    sig = lambda t : np.cos(2 * np.pi * t * p)
    sigs.append(sig(time))
    data += sig(time)

data_low_high = sigs[0] + sigs[2]
data_upper = sigs[1] + sigs[2]

freq = np.fft.rfftfreq(len(time), time[1] - time[0])
fft_abs = abs(np.fft.rfft(data))

# plt.plot(freq, fft_abs)
# plt.show()

order, lowcut, highcut = (4, 7, 15)

# ba output
# Bandpass filter ba
bp_b, bp_a = sp.signal.butter(order, [lowcut, highcut], btype='bandpass', fs=1001)
filt_data_bp_ba = sp.signal.lfilter(bp_b, bp_a, data)

# Bandstop filter ba 
bs_b, bs_a = sp.signal.butter(order, [lowcut, highcut], btype='bandstop', fs=1001)
filt_data_bs_ba = sp.signal.lfilter(bs_b, bs_a , data)

# Highpass filter ba
# Had to use filtfilt instead of lfilter. lfilter gave bad results
u_b, u_a = sp.signal.butter(order, lowcut, btype='highpass', fs=1001)
filt_data_u_ba = sp.signal.filtfilt(u_b, u_a, data)

# sos output
# Bandpass filter sos
sos_bp = sp.signal.butter(order, [lowcut, highcut], 'bandpass', fs=1001, output='sos')
filt_data_bp = sp.signal.sosfilt(sos_bp, data)

# Bandstop filter sos
sos_bs = sp.signal.butter(order, [lowcut, highcut], 'bandstop', fs=1001, output='sos')
filt_data_bs = sp.signal.sosfilt(sos_bs, data)

# Highpass filter sos
# Had to increase the order and decrease the lowcut
# sp.signal.sosfiltfilt gave betterresults that sosfilt
sos_u = sp.signal.butter(10, 6, 'highpass', fs=1001, output='sos')
filt_data_u = sp.signal.sosfiltfilt(sos_u, data)

# Plotting
fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)

ax1.plot(time, data)
ax1.set_title('Data : $\\cos(2 \\pi * 3t) + \\cos(2 \\pi * 10t) + \\cos(2 \\pi * 20t)$')

ax2.plot(time, data_low_high)
ax2.set_title('BS estimate : $\\cos(2 \\pi * 3t) + \\cos(2 \\pi * 20t)$')

ax3.plot(time, data_upper)
ax3.set_title('HP estimate : $\\cos(2 \\pi * 10t) + \\cos(2 \\pi * 20t)$')

ax4.plot(time, filt_data_bp_ba)
ax4.set_title(f'Bandpass filter\nlowcut = {lowcut}, highcut = {highcut} and order = {order}')

ax5.plot(time, filt_data_bs_ba)
ax5.set_title(f'Bandstop filter\nlowcut = {lowcut}, highcut = {highcut} and order = {order}')

ax6.plot(time, filt_data_u_ba)
ax6.set_title(f'Highpass filter\ncutoff = {lowcut} order=10')

fig.tight_layout()
plt.show()
