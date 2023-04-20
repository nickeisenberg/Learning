import imageio as io
import cv2
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import itertools as it
from skimage import filters

path = '/Users/nickeisenberg/GitRepos/Python_Notebook/ImageBlurring/'
img = io.imread(path + 'golden_retriever.jpg')

fig, ax = plt.subplots(1, 3)
ax[0].imshow(img[:,:,0])
ax[1].imshow(img[:,:,1])
ax[2].imshow(img[:,:,2])
plt.show()

def gaussian_bump_3d(mu, std, dim_len):
    dim_len = int((dim_len - 1) / 2)
    dom = np.arange(-dim_len, dim_len + 1, 1)
    pairs = it.product(dom, dom)
    surface = np.zeros((2 * dim_len + 1, 2 * dim_len + 1))
    for pair in pairs:
        indx = np.where(dom == pair[1])[0][0]
        indy = np.where(dom == pair[0])[0][0]
        r = np.sqrt(pair[0] ** 2 + pair[1] ** 2)
        surface[indy, indx] = norm(mu, std).pdf(r)
    return surface / surface.sum()

class Blur:

    def __init__(self, img):
        self.img = img

    def identity(self):
        kernel = np.zeros((3, 3))
        kernel[1][1] = 1
        return cv2.filter2D(src=self.img,
                            ddepth=-1,
                            kernel=kernel)

    def sharpen(self):
        kernel = np.zeros((3, 3))
        kernel[0][1] = -1
        kernel[1][np.array([0, 2])] = -1
        kernel[1][1] = 5
        kernel[2][1] = -1
        return cv2.filter2D(src=self.img,
                            ddepth=-1,
                            kernel=kernel)

    def gaussian(self, mu, std, dim_len, return_kernel=False):
        dim_len = int((dim_len - 1) / 2)
        dom = np.arange(-dim_len, dim_len + 1, 1)
        pairs = it.product(dom, dom)
        kernel = np.zeros((2 * dim_len + 1, 2 * dim_len + 1))
        for pair in pairs:
            indx = np.where(dom == pair[1])[0][0]
            indy = np.where(dom == pair[0])[0][0]
            r = np.sqrt(pair[0] ** 2 + pair[1] ** 2)
            kernel[indy, indx] = norm(mu, std).pdf(r)
        kernel = kernel / kernel.sum()

        if return_kernel:
            return cv2.filter2D(
                src=self.img,
                ddepth=-1,
                kernel=kernel), kernel
        else:
            return cv2.filter2D(
                src=self.img,
                ddepth=-1,
                kernel=kernel)

blurred_img, g_kernel = Blur(img).gaussian(0, 1, 5, return_kernel=True)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[1].imshow(blurred_img)
plt.show()

#-------------------------------------------------- 
def gaussian(mu, std, dim_len):
    dim_len = int((dim_len - 1) / 2)
    dom = np.arange(-dim_len, dim_len + 1, 1)
    pairs = it.product(dom, dom)
    kernel = np.zeros((2 * dim_len + 1, 2 * dim_len + 1))
    for pair in pairs:
        indx = np.where(dom == pair[1])[0][0]
        indy = np.where(dom == pair[0])[0][0]
        r = np.sqrt(pair[0] ** 2 + pair[1] ** 2)
        kernel[indy, indx] = norm(mu, std).pdf(r)
    # kernel = kernel / kernel.sum()
    return kernel

