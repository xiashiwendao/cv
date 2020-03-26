from keras.layers import Flatten, BatchNormalization, ZeroPadding2D, add
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, AveragePooling2D, GlobalAveragePooling2D
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import LeakyReLU
from keras.layers import concatenate
import cv2
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
from keras import utils
from keras.models import load_model
from util import *
import pdb
def get_vgg_model(dropout_rate=0):
    input_shape = (32, 32, 1)

    # input
    input_1 = Input(shape=input_shape)
    #import pdb;pdb.set_trace() 
    cv2d_1 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu")(input_1)
    cv2d_2 = Conv2D(filters=32, kernel_size=(3, 3), padding="same", activation="relu")(input_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cv2d_2)
    dropout_1 = Dropout(rate=dropout_rate)(pool_1)
    # flatten_1 = Flatten()(dropout_1)

    cv2d_3 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(dropout_1)
    cv2d_4 = Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu")(cv2d_3)
    pool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cv2d_4)
    dropout_2 = Dropout(rate=dropout_rate)(pool_2)
    # flatten_2 = Flatten()(dropout_2)

    cv2d_5 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(dropout_2)
    cv2d_6 = Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu")(cv2d_5)
    pool_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2))(cv2d_6)
    dropout_3 = Dropout(rate=dropout_rate)(pool_3)
    flatten_3 = Flatten()(dropout_3)

    dense_1 = Dense(units=256, activation="relu")(flatten_3)
    dropout_4 = Dropout(rate=dropout_rate)(dense_1)
    output = Dense(units=43, activation="softmax")(dropout_4)

    model = Model(inputs=input_1, outputs=output)
    # compile model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # summarize model
    model.summary()

    return model

def Conv2d_BN(x, nb_filter, kernel_size, strides=(1, 1), padding='same', name=None):
    if name is not None:
        bn_name = name + '_bn'
        conv_name = name + '_conv'
    else:
        bn_name = None
        conv_name = None

    x = Conv2D(nb_filter, kernel_size, padding=padding, strides=strides, activation='relu', name=conv_name)(x)
    x = BatchNormalization(axis=3, name=bn_name)(x)
    return x


def identity_Block(inpt, nb_filter, kernel_size, strides=(1, 1), with_conv_shortcut=False):
    x = Conv2d_BN(inpt, nb_filter=nb_filter, kernel_size=kernel_size, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=nb_filter, kernel_size=kernel_size, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=nb_filter, strides=strides, kernel_size=kernel_size)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


def bottleneck_Block(inpt, nb_filters, strides=(1, 1), with_conv_shortcut=False):
    k1, k2, k3 = nb_filters
    x = Conv2d_BN(inpt, nb_filter=k1, kernel_size=1, strides=strides, padding='same')
    x = Conv2d_BN(x, nb_filter=k2, kernel_size=3, padding='same')
    x = Conv2d_BN(x, nb_filter=k3, kernel_size=1, padding='same')
    if with_conv_shortcut:
        shortcut = Conv2d_BN(inpt, nb_filter=k3, strides=strides, kernel_size=1)
        x = add([x, shortcut])
        return x
    else:
        x = add([x, inpt])
        return x


def get_resnet_model(dropout_rate=0):
    classes = 43
    inpt = Input(shape=(32, 32, 1))
    x = ZeroPadding2D((3, 3))(inpt)

    # conv1
    x = Conv2d_BN(x, nb_filter=64, kernel_size=(3, 3), strides=(2, 2), padding='valid')
    x = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), padding='same')(x)

    # conv2_x
    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=64, kernel_size=(3, 3))

    # conv3_x
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=128, kernel_size=(3, 3))

    # conv4_x
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=256, kernel_size=(3, 3))

    # conv5_x
    x = identity_Block(x, nb_filter=512, kernel_size=(3, 3), strides=(2, 2), with_conv_shortcut=True)
    x = identity_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = identity_Block(x, nb_filter=512, kernel_size=(3, 3))
    x = AveragePooling2D(pool_size=(1, 1))(x)
    x = Flatten()(x)
    x = Dense(classes, activation='softmax')(x)

    model = Model(inputs=inpt, outputs=x)
    # compile model
    model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])
    # summarize model
    model.summary()
    return model

def DenseLayer(x, nb_filter, bn_size=4, alpha=0.0, drop_rate=0.2):

    # Bottleneck layers
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(bn_size*nb_filter, (1, 1), strides=(1,1), padding='same')(x)

    # Composite function
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(nb_filter, (3, 3), strides=(1,1), padding='same')(x)

    if drop_rate: x = Dropout(drop_rate)(x)

    return x

def DenseBlock(x, nb_layers, growth_rate, drop_rate=0.2):

    for ii in range(nb_layers):
        conv = DenseLayer(x, nb_filter=growth_rate, drop_rate=drop_rate)
        x = concatenate([x, conv], axis=3)

    return x

def TransitionLayer(x, compression=0.5, alpha=0.0, is_max=0):

    nb_filter = int(x.shape.as_list()[-1]*compression)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=alpha)(x)
    x = Conv2D(nb_filter, (1, 1), strides=(1,1), padding='same')(x)
    if is_max != 0: x = MaxPooling2D(pool_size=(2, 2), strides=2)(x)
    else: x = AveragePooling2D(pool_size=(2, 2), strides=2)(x)

    return x

growth_rate = 12

def get_densenet_model(dropout_rate=0):

    input_shape = (32, 32, 1)

    # input
    input_1 = Input(shape=input_shape)
    #import pdb;pdb.set_trace() 
    x = Conv2D(growth_rate*3, (3, 3), strides=1, padding='same')(input_1)
    x = BatchNormalization(axis=3)(x)
    x = LeakyReLU(alpha=0.1)(x)
    x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)

    x = TransitionLayer(x)

    x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)

    x = TransitionLayer(x)

    x = DenseBlock(x, 12, growth_rate, drop_rate=0.2)

    x = BatchNormalization(axis=3)(x)
    #x = GlobalAveragePooling2D()(x)
    x = Flatten()(x)
    x = Dense(units=256, activation="relu")(x)
    x = Dropout(rate=dropout_rate)(x)
    x = Dense(43, activation='softmax')(x)

    model = Model(input_1, x)
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.summary()
    return model