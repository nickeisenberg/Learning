import numpy as np

# convolution may not be usefull
# defining a circ_conv like circ_cross_conv is probably better
def convolution(x, y, t):
    nx = len(x)
    ny = len(y)

    summand = []
    for n in range(nx):
        if t - n < 0:
            summand.append(0)
        elif t - n >= ny:
            summand.append(0)
        else:
            summand.append(x[n] * y[t - n])
    summand = np.array(summand)

    conv = sum(summand)
    return conv

# Unsure is cross_corr has significant meaning.
# circ_cross_corr may be a more useful function
def cross_corr(x, y, t, real=True):
    nx = len(x)
    ny = len(y)
    summand = []

    if not real:
        x = np.conj(x)

    for n in range(nx):
        if t + n < 0:
            summand.append(0)
        elif t + n >= ny:
            summand.append(0)
        else:
            summand.append(x[n] * y[t + n])

    summand = np.array(summand)
    cc = sum(summand)

    return cc

def circ_cross_corr(x, y, t, real=True):
    nx = len(x)
    ny = len(y)
    summand = []

    if not real:
        x = np.conj(x)

    for n in range(nx):
        if t + n < -ny:
            if (t+n) % ny == 0:
                sumand.append(x[n] * y[-ny])
            else:
                index = (t + n) % ny
                sumand.append(x[n] * y[index])
        elif t + n > ny - 1:
            if (t + n) % (ny - 1) == 0:
                summand.append(x[n] * y[ny - 1])
            else:
                index = ((t + n) % (ny - 1)) - 1
                summand.append(x[n] * y[index])
        else:
            summand.append(x[n] * y[t + n])

    summand = np.array(summand)
    cc = sum(summand)

    return cc

if __name__ == '__main__':

    x = [1, 2, 3, 4, 5, 5]
    y = [1, 2, 3, 4, 4, 5]
    cc = circular_convolution(x, y, 3)
    print(x)
    print(y)
    print(cc)


