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

img_b, g_kernel = Blur(img).gaussian(0, 1, 5, return_kernel=True)

fig, ax = plt.subplots(1, 2)
ax[0].imshow(img)
ax[1].imshow(img_b)
plt.show()

#-------------------------------------------------- 
# The following shows that when blurring the image, the convolution happens
# channel by channel.
img0 = img[:,:,0]
img1 = img[:,:,1]
img2 = img[:,:,2]

img0_b = Blur(img0).gaussian(0, 1, 5)
img1_b = Blur(img1).gaussian(0, 1, 5)
img2_b = Blur(img2).gaussian(0, 1, 5)

img_b_ = np.dstack((img0_b, img1_b, img2_b))

print((img_b - img_b_).min())
print((img_b - img_b_).max())
#-------------------------------------------------- 
# Some testing with the fft and ifft

x = np.linspace(0, 1, 1000)
data = np.sin(2 * np.pi * x * 5)
data += np.sin(2 * np.pi * x * 10)
data += np.sin(2 * np.pi * x * 15)

fft_data = np.fft.rfft(data)

freqs = np.fft.rfftfreq(data.size)

freqs.shape

plt.plot(freqs, np.abs(fft_data))
plt.show()

plt.plot(np.fft.irfft(fft_data))
plt.plot(data)
plt.show()
#-------------------------------------------------- 

data = np.random.normal(size=(1000, 1000, 3))

fft_data = np.fft.fftn(data)

fft_data.shape

(data - np.fft.ifftn(fft_data).real).min()

#-------------------------------------------------- 

# Estimating the blur kernal
# img * blur_ker = img_blur

# Doing this channel by channel
# It is import to round when computing the ifft.
# If we dont rounf then there will be e-15 entries instead of 0.
# This causes minor problems with the image reconstruction
kers_fft = []
est_blur_image = []
for channel in range(img.shape[-1]):
    imgc = img[:,:, channel]
    imgc_b = Blur(imgc).gaussian(0, 1, 5)
    imgc_fft = np.fft.fft2(imgc)
    imgc_b_fft = np.fft.fft2(imgc_b)
    ker = imgc_b_fft / imgc_fft
    est_blur_image.append(
        np.fft.ifft2(imgc_fft * ker).real.round(5)
    )
    kers_fft.append(ker)

est_blur_image = np.dstack(est_blur_image).astype(int)
err = est_blur_image - img_b
np.sum([err != 0])

#-------------------------------------------------- 

# We don not need to do this channel by channel and we can use fftn instead.
ker_fft = np.fft.fftn(img_b) / np.fft.fftn(img)
ker_est = np.fft.ifftn(ker_fft).real
est_blur_image = np.fft.ifftn(ker_fft * np.fft.fftn(img)).real.round(5)

err = est_blur_image - img_b
np.sum([err != 0])

# We can also use the ker_est and convolve the original image to reconstruct
# the blurred image
est_blur_image_2 = cv2.filter2D(
    src=img, ddepth=-1, kernel=ker_est)

err2 = est_blur_image - img_b
np.sum([err2 != 0])

#-------------------------------------------------- 

def gaussian(mu, std, dim_len, return_grid=False):
    dim_len = int((dim_len - 1) / 2)
    dom = np.arange(-dim_len, dim_len + 1, 1) + mu
    X, Y = np.meshgrid(dom, dom)
    r = np.sqrt((X - mu) ** 2 + (Y - mu) ** 2)
    surface = (1 / np.sqrt(2 * np.pi * std ** 2)) * np.exp(-(r ** 2) / (2 * std))
    if return_grid:
        return X, Y, surface
    else:
        return surface


X, Y, Z = gaussian(0, 1, 21, return_grid=True)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1, projection='3d')
ax.plot_surface(X, Y, Z)
plt.show()

