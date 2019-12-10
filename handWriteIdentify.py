import pandas as pd
import numpy as np
from keras.datasets import mnist
from keras.models import Model
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPool2D
from keras.layers.normalization import BatchNormalization
from matplotlib import pyplot as plt
# 获取数据集
(X_train, y_train),(X_test, y_test) = mnist.load_data()
print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
# 探索数据-展示样本
for i in range(9):
    plt.subplot(3, 3, i+1)
    plt.imshow(X_train[i], cmap="gray", interpolation="none")

plt.show()
# 数据探索-数据集划分
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1)
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1)
print("X_train: ", X_train.shape)
print("X_test: ", X_test.shape)
# 数据探索-分析数据集
import seaborn as sns
sns.countplot(y_train)
plt.show()
# 特征工程-独热编码
from keras.utils import np_utils
Y_train = np_utils.to_categorical(y_train)
Y_train.shape
Y_test = np_utils.to_categorical(y_test)
Y_test.shape
# 构建模型
from keras.models import Sequential
input_shape=(28, 28, 1)
nb_classes = 10

model = Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal", input_shape=input_shape))
model.add(Conv2D(32, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal"))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.2))

model.add(Conv2D(64, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal"))
model.add(Conv2D(64, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal"))
model.add(MaxPool2D(2,2))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3,3), activation="relu", kernel_initializer="he_normal"))
model.add(Dropout(0.25))
model.add(Flatten())

model.add(Dense(128, activation="relu"))
model.add(BatchNormalization())
model.add(Dropout(0.25))
model.add(Dense(nb_classes, activation="softmax"))

from keras.optimizers import Adam
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator

model.compile(optimizer=Adam(), loss=categorical_crossentropy, metrics=["accuracy"])
model.summary()
datagen = ImageDataGenerator(featurewise_center=True, featurewise_std_normalization=True)
datagen.fit(X_train)

# 训练模型
from keras.callbacks import ModelCheckpoint
filePath = "model.hdf5"
checkpoint = ModelCheckpoint(filePath, save_best_only=True, monitor="val_loss")
h = model.fit_generator(datagen.flow(X_train, Y_train, batch_size=1000), \
    steps_per_epoch=len(X_train)/1000,epochs=3, validation_data=datagen.flow(X_test, Y_test, batch_size=len(X_test)), validation_steps=1, callbacks=[checkpoint])

# 评估模型-基于趋势图
h.history()
# 下面就是画出来train以及validation趋势图，看到的是由低走到高
# 省略... ...

# 评估模型-基于测试集的score
score = model.evaluate(X_test, Y_test)
print("test score", score[0])
print("Test accuracy", score[1]) # 损失函数值

# 评估模型-基于抽样分析预测测试集结果
predicted_classed= model.predict_classes(X_test)
correct_indices = np.nonzero(predicted_classed== Y_test)[0]
incorrect_indices = np.nonzero(predicted_classed != Y_test)[0]
plt.figure()
for i, correct in enumerate(correct_indices[:9]):
    plt.subplot(3,3,i + 1)
    plt.imshow(X_test[correct].reshape(28, 28), cmap="gray", interpolation=False)
    plt.title("Predict Class: {}; Accuracy Class {}", predicted_classed[correct], X_test[correct])