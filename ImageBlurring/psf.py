import matplotlib.pyplot as plt
import numpy as np
import imageio as io
import cv2

img_path = '/Users/nickeisenberg/GitRepos/Python_Notebook/ImageBlurring'
img_path += '/golden_retriever.jpg'

img = io.imread(img_path)

img = img[1:,:,:]

x = np.arange(-img.shape[0] / 2, img.shape[0] / 2, 1)

X, Y = np.meshgrid(x, x)

Z = X ** 2 + Y ** 2

psf = np.exp(- Z / 10)

plt.imshow(psf)
plt.show()

blur = cv2.filter2D(img, kernel=psf, ddepth=-1)

plt.imshow(img)
plt.show()

plt.imshow(blur)
plt.show()

identity = np.array([
    [0, 0, 0],
    [0, 1, 0],
    [0, 0, 0]
])


blur = cv2.filter2D(img, kernel=identity, ddepth=-1)

np.max(blur - img)
