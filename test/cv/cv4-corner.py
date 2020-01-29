import cv2
import numpy as np
import os,sys
from matplotlib import pyplot as plt

file = 'test{}dataset{}chessboard.png'.format(os.sep, os.sep, os.sep)
img_raw = cv2.imread(file)
img_raw_float = np.float32(img_raw) # must convert to float, or can't be display by plot tool

# why need to conver to gray
img_gray = cv2.cvtColor(img_raw_float, cv2.COLOR_BGR2GRAY) 
img_gray_float = np.float32(img_gray)

fig = plt.figure(figsize=(12, 12))
ax = fig.add_subplot(121)
#ax.set_title('img_raw')
plt.imshow(img_raw_float)

fig.add_subplot(122)
plt.imshow(img_gray_float)
plt.show()

dst = cv2.cornerHarris(img_gray_float, 2, 3, 0.04)
dst_dilate = cv2.dilate(dst, None) # used to extend /strength the effective of corner
img[dst_dilate>0.01*dst_dilate.max()] = [0, 0, 255]
plt.imshow(img)
plt.show()
