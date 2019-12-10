import numpy as np
import sys, os
from matplotlib import pyplot as plt
from matplotlib import image as mpimg

print(os.getcwd())
img_path = 'dataset/img/dog.jpg'
img1 = mpimg.imread(img_path)
plt.imshow(img1)
plt.show()
print(np.shape(img1))
type(img1) # type is nparray

import cv2
img2 = cv2.imread(img_path)
plt.imshow(img2)
plt.show()
type(img2) # type is nparray

img2_cvt = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
plt.imshow(img2_cvt)
plt.show()

# use PIL to operation image
from PIL import Image
img3 = Image.open(img_path)
plt.imshow(img3)
plt.show()
type(img3) # type is PIL.JpegImagePlugin.JpegImageFile

# image => nparray
img3_arr = np.array(img3)
plt.imshow(img3_arr)
plt.show()
type(img3_arr)

# save the middle result
np.save('pic.npy', img3_arr)
img3_load = np.load('pic.npy')
plt.imshow(img3_load)
plt.show()

# scaled the image(figure will be weired)
target_size = (200, 200)
img3_scaled = img3.resize(target_size)
print(img3_scaled.size)

# better handle for resize
img3_copy = img3.copy()
img3_copy.thumbnail(target_size, Image.ANTIALIAS)
print('imge3_copy size:', img3_copy.size, '; image3 size: ', img3.size)

# use the paste the image to a white image
new_img = Image.new('RGB', img3.size, (255, 255, 255))
new_img.paste(img3_copy, (int(target_size[0] - img3_copy.size[0]), int(target_size[1] - img3_copy.size[1])))
print('new image postition: ', (int(target_size[0] - img3_copy.size[0]), int(target_size[1] - img3_copy.size[1])))

fig = plt.figure(figsize=(12, 12))
a = fig.add_subplot(2, 2, 1)
plt.imshow(img3)
a.set_title('original')

a = fig.add_subplot(2, 2, 2)
plt.imshow(img3_scaled)
a.set_title('direct scaled')

a = fig.add_subplot(2, 2, 3)
plt.imshow(img3_copy)
a.set_title('better scaled')

a = fig.add_subplot(2, 2, 4)
plt.imshow(new_img)
a.set_title('new_img')


plt.show()



