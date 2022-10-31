import numpy as np
import matplotlib.pyplot as plt
from math import ceil, floor
import scipy as sp
from scipy import signal

x = np.linspace(1, 15, 1000)
curve = lambda x : x / 5 + np.cos(2 * x)
y = curve(x)
plt.plot(x,y)

peaks, _ = sp.signal.find_peaks(y)
x_peak = []
y_peak = []
for peak in peaks:
    x_peak.append(x[peak])
    y_peak.append(y[peak])
x_peak, y_peak = (np.array(x_peak), np.array(y_peak))
plt.scatter(x_peak, y_peak, marker='x', c='red')

# half = sp.signal.peak_widths(y, peaks, rel_height=.5)[0]
endpoints = sp.signal.peak_widths(y, peaks, rel_height=1)[2:]
end_l = endpoints[0]
end_r = endpoints[1]
for l, r in zip(end_l, end_r):
    xls, xlb = (x[floor(l)], x[ceil(l)])
    xrs, xrb = (x[floor(r)], x[ceil(r)])
    delta_l = xlb - xls
    delta_r = xrb - xrs
    xl = xls + delta_l * (l - int(l))
    xr = xrs + delta_r * (r - int(r))
    plt.plot([xl, xr], curve(np.array([xl, xr])), c='red')

plt.title('sp.find_peak example')
plt.show()
