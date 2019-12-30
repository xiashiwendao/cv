from PIL import Image
from matplotlib import pyplot as plt
import numpy as np

# load the picture
img3 = Image.open('dataset\\image_test\\dark.jpg')
plt.imshow(img3)
plt.show()

# convert the Image to numpy, for later display
img3 = np.array(img3)
img3.dtype

# show hist graph, NOTE: you need to use the ravel method.
plt.hist(img3.ravel())
plt.show()

# show the CDF(Cumulative Distribution Function) graph, which show the cumulative trends
plt.hist(img3.ravel(), bins=255, cumulative=True)
plt.show()

from PIL import ImageOps
# ? tell me what is the difference from Image.fromarray(img3) to numpy(img3)
img_eq = ImageOps.equalize(Image.fromarray(img3))
fig = plt.figure(figsize=(12, 6))
a = fig.add_subplot(1,2,1)
a.set_title("before")
plt.imshow(img3)

a = fig.add_subplot(1, 2, 2)
a.set_title("after")
plt.imshow(img_eq)

plt.show()

# see the hist of the picture after equalized 
img_arr_eq = np.array(img_eq)
fig = plt.figure(figsize=(12,6))
a = fig.add_subplot(2,2,1)
a.set_title('img raw')
plt.hist(img3.ravel())

a = fig.add_subplot(2,2,2)
a.set_title('equalized img')
plt.hist(img_arr_eq.ravel())

a = fig.add_subplot(2,2,3)
a.set_title('img raw cumulative')
plt.hist(img3.ravel(), bins=255, cumulative=True)

a = fig.add_subplot(2,2,4)
a.set_title('img equlized CDF')
plt.hist(img_arr_eq.ravel(), bins=255, cumulative=True)

plt.show()
