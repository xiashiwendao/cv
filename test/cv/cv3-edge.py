from scipy import ndimage
import skimage.color as sc
import numpy as np

from skimage.feature import canny
from matplotlib import pyplot as plt

def edge_sobel(image):
    image = sc.rgb2gray(image)
    dx = ndimage.sobel(image, 1)
    dy = ndimage.sobel(image, 0)
    mag = np.hypot(dx, dy)
    mag *= 255.0 / np.amax(mag)
    mag = mag.astype(np.uint8)

    return mag

def showTransformatedGraphics(image_raw, image_noise,image_rotate, image_filter):
    
    fig = plt.figure(figsize=(14,14))
    ax = fig.add_subplot(141)
    ax.set_axis_off()
    ax.set_title('raw image')
    plt.imshow(image_raw)

    ax = fig.add_subplot(142)
    ax.set_axis_off()
    ax.set_title('noise image')
    plt.imshow(image_noise)

    ax = fig.add_subplot(143)
    ax.set_axis_off()
    ax.set_title('rotate image')
    plt.imshow(image_rotate)

    ax = fig.add_subplot(144)
    ax.set_axis_off()
    ax.set_title('gauss filter image')
    plt.imshow(image_filter)

    plt.show()

def showEdge(image_filter, sx, sy, sob, canny_edge):
    # draw the five image
    fig = plt.figure(figsize=(14, 14))
    ax = fig.add_subplot(151)
    ax.set_axis_off()
    ax.set_title('square')
    ax.imshow(image_filter)

    ax = fig.add_subplot(152)
    ax.set_axis_off()
    ax.set_title('sobel (x)')
    ax.imshow(sx)

    ax = fig.add_subplot(153)
    ax.set_axis_off()
    ax.set_title('sobel (y)')
    ax.imshow(sy)

    ax = fig.add_subplot(154)
    ax.set_axis_off()
    ax.set_title('sobel edge')
    ax.imshow(sob)

    ax = fig.add_subplot(155)
    ax.set_axis_off()
    ax.set_title('canny edge')
    ax.imshow(canny_edge)

    plt.show()

# draw a tangle
image_raw = np.zeros((256, 256))
image_raw[64:-64, 64:-64] = 1
image_raw[96:-96, 0:25] = 1
# add some noise on the tangle
noise = np.random.randn(image_raw.shape[0], image_raw.shape[1]) /2
image_noise = image_raw + noise

fig = plt.figure(figsize=(12,12))
ax = fig.add_subplot(121)
plt.imshow(image_noise)

ax = fig.add_subplot(122)
plt.imshow(edge_sobel(image_noise))

plt.show()
# to rotate the image and add filter
image_rotate = ndimage.rotate(image_noise, 15, mode='constant')
image_filter = ndimage.gaussian_filter(image_rotate,  8)

showTransformatedGraphics(image_raw, image_noise,image_rotate, image_filter)

# use the sobel to detect edge
sx = ndimage.sobel(image_filter, axis=0, mode='constant')
sy = ndimage.sobel(image_filter, axis=1, mode='constant')
sob = np.hypot(sx, sy) # thru the numpy's hypot function to merge the two matrix to one

# use the canny to detect edge
canny_edge = canny(image_filter, sigma=5)


# draw the five image
showEdge(image_filter, sx, sy, sob, canny_edge)
