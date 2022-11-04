import numpy as np

x = np.array([1,4,6])
x_n = (x - np.mean(x)) / np.sqrt(np.dot(x, x))
y = np.array([1,2,3])
y_n = (y - np.mean(y)) / np.sqrt(np.dot(y, y))

def pearson_corr(x, y):
    ux, uy = np.mean(x), np.mean(y)
    x_cen, y_cen = x - ux, y - uy
    return np.dot(x_cen, y_cen) / (np.sqrt(np.sum(np.square(x_cen))) * np.sqrt(np.sum(np.square(y_cen))))

print(np.corrcoef(np.array([x,y])))
