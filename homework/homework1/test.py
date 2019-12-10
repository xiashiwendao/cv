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

arr = np.array([[[[ 46., 110., 174.],
         [ 45., 108., 177.],
         [ 46., 109., 176.],
         [ 63., 117., 164.],
         [ 59., 117., 163.],
         [ 61., 120., 164.]],

        [[ 41., 106., 170.],
         [ 40., 104., 175.],
         [ 40., 105., 173.],
         [ 61., 111., 162.],
         [ 63., 110., 162.],
         [ 65., 111., 163.]]]])
arr[:,:,:,0]
arr_reshape = arr.reshape([arr.shape[0]*arr.shape[1]*arr.shape[2],3])
np.mean(arr_reshape, axis=0)

from keras.preprocessing.text import Tokenizer

tokenizer = Tokenizer()
lines = ['this is good', 'this is a cat']
tokenizer.fit_on_texts(lines)

results = tokenizer.texts_to_sequences(['cat is good'])
print(results)

from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

max_length = 6
vocab_size = 661
seq = [2, 660, 6, 229, 3]
i = 1
in_seq, out_seq = seq[:i], seq[i]
in_seq = pad_sequences([in_seq], maxlen=max_length)[0]
out_seq = to_categorical([out_seq], num_classes=vocab_size)[0]

print(in_seq)
print(out_seq)
