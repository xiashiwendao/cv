from keras.models import model_from_json
from PIL import Image as pil_image
from keras import backend as K
import numpy as np
from pickle import dump
from os import listdir
from keras.models import Model
import keras
import sys,os


def load_vgg16_model():
    """从当前目录下面的 vgg16_exported.json 和 vgg16_exported.h5 两个文件中导入 VGG16 网络并返回创建的网络模型
    # Returns
        创建的网络模型 model
    """
    json_file = open(os.path.join("dataset\\vgg16\\", "vgg16_exported.json"))
    json = json_file.read()
    json_file.close()

    model = model_from_json(json)
    model.load_weights(os.path.join("dataset\\vgg16\\", "vgg16_exported.h5"))

    return model


def preprocess_input(x):
    """预处理图像用于网络输入, 将图像由RGB格式转为BGR格式.
       将图像的每一个图像通道减去其均值

    # Arguments
        x: numpy 数组, 4维.
        data_format: Data format of the image array.

    # Returns
        Preprocessed Numpy array.
    """
    # 预处理图像用于网络输入, 将图像由RGB格式转为BGR格式
    for oneDim in x:
        for twoDim in oneDim:
            index = 0
            for rgb in twoDim:
                # 减去对应的均值（这里的均值是vgg16全体图像的均值，非单张图片的均值）
                twoDim[index] = np.array([rgb[2]-103.939, rgb[1]-116.779, rgb[0]-123.68])
                index+=1

    return x



def load_img_as_np_array(path, target_size):
    """从给定文件加载图像,转换图像大小为给定target_size,返回32位浮点数numpy数组.
    
    # Arguments
        path: 图像文件路径
        target_size: 元组(图像高度, 图像宽度).

    # Returns
        A PIL Image instance.
    """
    img = pil_image.open(path)
    img_resize = img.resize(target_size, pil_image.NEAREST)
    
    return np.asarray(img_resize, dtype=K.floatx())


def extract_features(directory):
    # drop out the last layer to get 4096 output(get a new Model)
    # for we just want to get the feature, but not the classifiers
    # the last layer just base on the features to get classifiers
    model = load_vgg16_model()
    print("+++++ pop ++++")
    model.layers.pop()
    print("+++++ Model ++++")
    model = Model(input=model.inputs, output=model.layers[-1].output)
    features = dict()
    print("+++++ extract the feature ++++")
    # extract the feature
    for fn in os.listdir(directory):
        filename = os.path.join(directory, fn)
        print("++++ file name is: ", filename, " +++++++++")
        img_array = load_img_as_np_array(filename, target_size=(224, 224))
        img_array = img_array.reshape((1, img_array.shape[0],img_array.shape[1],img_array.shape[2]))
        img_array = preprocess_input(img_array)
        print("++++ img_array is: ", np.shape(img_array), " +++++++++")
        feature = model.predict(img_array, verbose=0)

        id = fn.split('.')[0]
        features[id] = feature
    
    return features


# if __name__ == '__main__':
# 提取所有图像的特征，保存在一个文件中, 大约一小时的时间，最后的文件大小为127M
directory = 'dataset\\vgg16\\image_test'
filepath = os.path.join('dataset\\vgg16\\', "vgg16_exported.json")
os.path.abspath(filepath)
os.getcwd()
os.path.exists(filepath)
features = extract_features(directory)
print('提取特征的文件个数：%d' % len(features))
print(keras.backend.image_data_format())
#保存特征到文件
dump(features, open('features.pkl', 'wb'))


aa = np.array([1,2,3])
aa.shape()