import yfinance as yf
import matplotlib.pyplot as plt
import numpy as np

spy_info = yf.Ticker('SPY')

price_history = spy_info.history(period='5y',
                                 interval='1d',
                                 actions=False)


spy_open = price_history['Open'].values[:500]
time = np.linspace(0, 5, len(spy_open))

freq = np.fft.rfftfreq(len(spy_open))
fft = abs(np.fft.rfft(spy_open))

# plt.plot(time, spy_open)
plt.plot(freq[1:], fft[1:])
plt.show()


