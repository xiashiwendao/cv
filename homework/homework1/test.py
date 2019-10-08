from  PIL import Image as pil_image
import numpy as np
from keras import backend as K

path = "C:\\Users\\wenyang.zhang\\Documents\\MySpace\\practice\\github\\cv\\dataset\\image_test\\42637986_135a9786a6.jpg"
img = pil_image.open(path)
img.size
img = img.resize((224, 224), pil_image.NEAREST)

img_arr = np.asarray(img, dtype=K.floatx())
img_arr.shape
img_arr = img_arr.reshape(1, img_arr.shape[0], img_arr.shape[1], img_arr.shape[2])
img_arr_rgb = img_arr[3]
img_arr_rgb
print("未处理前：")
img_arr[:,:,:]
last_element_before = []
last_element_after = []
for oneDim in img_arr:
    for twoDim in oneDim:
        index = 0
        for rgb in twoDim:
            last_element_before = rgb.copy()
            rgb = np.array([rgb[2], rgb[1], rgb[0]])
            last_element_after = twoDim[index]
            index+=1
print("交换后：")
img_arr[:,:,:]
print("最后处理的数据（前）：", last_element_before)
print("最后处理的数据（后）：", last_element_after)