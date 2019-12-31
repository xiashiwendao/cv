import skimage
from PIL import Image
from PIL import ImageOps
from matplotlib import pyplot as plt
import numpy as np

img3 = Image.open('dataset\\image_test\\dog.jpg')
# plt.imshow(img3)

img3_arr = np.array(img3)
plt.imshow(img3)
plt.show()

img3_noise = skimage.util.random_noise(img3_arr)
plt.imshow(img3_noise)
plt.show()

fig = plt.figure(figsize=(12,4))
fig.add_subplot(1,2,1)
plt.imshow(img3)
fig.add_subplot(1,2,2)
plt.imshow(img3_noise)
plt.show()

# GAUSS Filter, picuture will become blur
from scipy.ndimage.filters import gaussian_filter as gauss
img3_gauss = gauss(img3_noise, sigma=1) # ? TODO sigma mean, control the blur strength

fig = plt.figure(figsize=(12,4))
fig.add_subplot(2,2,1)
plt.imshow(img3)
fig.add_subplot(2,2,3)
plt.imshow(img3_noise)
fig.add_subplot(2,2,4)
plt.imshow(img3_gauss)

plt.show()

# the media filter, picuture will more sharp
from scipy.ndimage import filters
img3_med = filters.median_filter(img3_noise, size=3)

fig = plt.figure(figsize=(12,4))
fig.add_subplot(2,2,1)
plt.imshow(img3)

fig.add_subplot(2,2,2)
plt.imshow(img3_noise)

fig.add_subplot(2,2,3)
plt.imshow(img3_gauss)

fig.add_subplot(2,2,4)
plt.imshow(img3_med)

plt.show()
