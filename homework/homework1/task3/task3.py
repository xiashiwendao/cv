from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from numpy import array
from pickle import load
import numpy as np
import homework.homework1.task3
import homework.homework1.task3.util as util
import sys, os
from keras.preprocessing.text import Tokenizer

def create_tokenizer():

    """
    根据训练数据集中图像名，和其对应的标题，生成一个tokenizer,作为LSTM的输入/输出必须是数字，所以需要我们使用
    字典数据类型来存储文字和数字对应关系。
    :return: 生成的tokenizer
    https://keras-cn.readthedocs.io/en/latest/legacy/preprocessing/text/#tokenizer
    """

    train_image_names = util.load_image_names('homework\\homework1\\task3\\Flickr_8k.trainImages.txt')
    train_descriptions = util.load_clean_captions('homework\\homework1\\task3\\descriptions.txt', train_image_names)
    lines = util.to_list(train_descriptions)

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(lines)

    return tokenizer


def test_create_token():
    tokenizer = Tokenizer()
    lines = ['this is good', 'that is a cat']
    tokenizer.fit_on_texts(lines)

    results = tokenizer.texts_to_sequences(['cat is good'])
    print(results)
    # you will see the last three word (they are the same) has the same values for the two results
    results = tokenizer.texts_to_sequences(['that cat is good'])
    print(results)


def create_input_data_for_one_image(seq, photo_feature, max_length, vocab_size):
    """
    从输入的一张图片的标题（已将英文单词转换为整数）和图片特征构造一组输入
    :param seq: 图片的标题（已将英文单词转换为整数）序列
    :param photo_feature: 图像的特征numpy数组
    :param max_length: 训练数据集中最长的标题的长度
    :param vocab_size: 训练集中所有图像标题的单词数量
    :return: tuple:
            第一个元素为 list, list的元素为图像的特征
            第二个元素为 list, list的元素为图像标题的前缀
            第三个元素为 list, list的元素为图像标题的下一个单词(根据图像特征和标题的前缀产生)的独热编码

    https://keras.io/utils/ to_categorical

        Examples:
            from pickle import load
            import numpy as np
            tokenizer = load(open('tokenizer.pkl', 'rb'))
            desc = 'startseq cat on table endseq'
            seq = tokenizer.texts_to_sequences([desc])[0]
            print(seq)
            [2, 660, 6, 229, 3]
            photo_feature = np.array([0.345, 0.57, 0.003, 0.987])
            input1, input2, output = task3.create_input_data_for_one_image(seq, photo_feature, 6, 661)
            print(input1)
            [array([0.345, 0.57 , 0.003, 0.987]), array([0.345, 0.57 , 0.003, 0.987]), array([0.345, 0.57 , 0.003, 0.987]), array([0.345, 0.57 , 0.003, 0.987]), array([0.345, 0.57 , 0.003, 0.987])]
            print(input2)
            [array([0, 0, 0, 0, 0, 2], dtype=int32), array([  0,   0,   0,   0,   2, 660], dtype=int32), array([  0,   0,   0,   2, 660,   6], dtype=int32), array([  0,   0,   2, 660,   6, 229], dtype=int32)]
            print(output[0])
            array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
                  ...
                   0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 1.],
                  dtype=float32)
        """
    #seq, photo_feature, max_length, vocab_size
    input1 = list()
    input2 = list()
    output = list()
    seq_len = len(seq)
    for i in range(seq_len - 1):
        index = i + 1 # from i = 1, so need to add one
        in_seq, out_seq = seq[:index], seq[index]
        in_seq_padded = pad_sequences([in_seq], maxlen=max_length)[0]
        out_seq_cat = to_categorical([out_seq], num_classes=vocab_size)

        input1.append(photo_feature) # feature will always be the same
        input2.append(in_seq_padded) # surfix of the title(every time will add a word)
        output.append(out_seq_cat) # the word follow the surfix word(s)
    
    return input1, input2, output


def create_input_data(tokenizer, max_length, descriptions, photos_features, vocab_size):
    """
    从输入的图片标题list和图片特征构造一组输入

    Args:
    :param tokenizer: 英文单词和整数转换的工具keras.preprocessing.text.Tokenizer
    :param max_length: 训练数据集中最长的标题的长度
    :param descriptions: dict, key 为图像的名(不带.jpg后缀), value 为list, 包含一个图像的几个不同的描述
    :param photos_features:  dict, key 为图像的名(不带.jpg后缀), value 为numpy array 图像的特征
    :param vocab_size: 训练集中表的单词数量
    :return: tuple:
            第一个元素为 numpy array, 元素为图像的特征, 它本身也是 numpy.array
            第二个元素为 numpy array, 元素为图像标题的前缀, 它自身也是 numpy.array
            第三个元素为 numpy array, 元素为图像标题的下一个单词(根据图像特征和标题的前缀产生) 也为numpy.array

    Examples:
        from pickle import load
        tokenizer = load(open('tokenizer.pkl', 'rb'))
        max_length = 6
        descriptions = {'1235345':['startseq one bird on tree endseq', "startseq red bird on tree endseq"],
                        '1234546':['startseq one boy play water endseq', "startseq one boy run across water endseq"]}
        photo_features = {'1235345':[ 0.434,  0.534,  0.212,  0.98 ],
                          '1234546':[ 0.534,  0.634,  0.712,  0.28 ]}
        vocab_size = 7378
        print(create_input_data(tokenizer, max_length, descriptions, photo_features, vocab_size))
            (array([[ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.434,  0.534,  0.212,  0.98 ],
                   [ 0.534,  0.634,  0.712,  0.28 ],
                   [ 0.534,  0.634,  0.712,  0.28 ],
                   [ 0.534,  0.634,  0.712,  0.28 ],
                   [ 0.534,  0.634,  0.712,  0.28 ],
                   [ 0.534,  0.634,  0.712,  0.28 ],
                   [ 0.534,  0.634,  0.712,  0.28 ],
                   [ 0.534,  0.634,  0.712,  0.28 ],
                   [ 0.534,  0.634,  0.712,  0.28 ],
                   [ 0.534,  0.634,  0.712,  0.28 ],
                   [ 0.534,  0.634,  0.712,  0.28 ],
                   [ 0.534,  0.634,  0.712,  0.28 ]]),
            array([[  0,   0,   0,   0,   0,   2],
                   [  0,   0,   0,   0,   2,  59],
                   [  0,   0,   0,   2,  59, 254],
                   [  0,   0,   2,  59, 254,   6],
                   [  0,   2,  59, 254,   6, 134],
                   [  0,   0,   0,   0,   0,   2],
                   [  0,   0,   0,   0,   2,  26],
                   [  0,   0,   0,   2,  26, 254],
                   [  0,   0,   2,  26, 254,   6],
                   [  0,   2,  26, 254,   6, 134],
                   [  0,   0,   0,   0,   0,   2],
                   [  0,   0,   0,   0,   2,  59],
                   [  0,   0,   0,   2,  59,  16],
                   [  0,   0,   2,  59,  16,  82],
                   [  0,   2,  59,  16,  82,  24],
                   [  0,   0,   0,   0,   0,   2],
                   [  0,   0,   0,   0,   2,  59],
                   [  0,   0,   0,   2,  59,  16],
                   [  0,   0,   2,  59,  16, 165],
                   [  0,   2,  59,  16, 165, 127],
                   [  2,  59,  16, 165, 127,  24]]),
            array([[ 0.,  0.,  0., ...,  0.,  0.,  0.],
                   [ 0.,  0.,  0., ...,  0.,  0.,  0.],
                   [ 0.,  0.,  0., ...,  0.,  0.,  0.],
                   ...,
                   [ 0.,  0.,  0., ...,  0.,  0.,  0.],
                   [ 0.,  0.,  0., ...,  0.,  0.,  0.],
                   [ 0.,  0.,  0., ...,  0.,  0.,  0.]]))
    """
    X1, X2, y = list(), list(), list()
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            seq = tokenizer.texts_to_sequences([desc])[0]
            photo_feature = photos_features[key][0]
            input1, input2, output = create_input_data_for_one_image(seq, photo_feature, max_length, vocab_size)
            X1.append(input1)
            X2.append(input2)
            y.append(output)

    return array(X1), array(X2), array(y) 


from pickle import load
import numpy as np
tokenizer = load(open('homework\\homework1\\task3\\tokenizer.pkl', 'rb'))
desc = 'startseq cat on table endseq'
seq = tokenizer.texts_to_sequences([desc])[0]
print(seq)
#[2, 660, 6, 229, 3]
photo_feature = np.array([0.345, 0.57, 0.003, 0.987])
input1, input2, output = create_input_data_for_one_image(seq, photo_feature, 6, 661)
print(input1)
#[array([0.345, 0.57 , 0.003, 0.987]), array([0.345, 0.57 , 0.003, 0.987]), array([0.345, 0.57 , 0.003, 0.987]), array([0.345, 0.57 , 0.003, 0.987]), array([0.345, 0.57 , 0.003, 0.987])]
print(input2)
#[array([0, 0, 0, 0, 0, 2], dtype=int32), array([  0,   0,   0,   0,   2, 660], dtype=int32), array([  0,   0,   0,   2, 660,   6], dtype=int32), array([  0,   0,   2, 660,   6, 229], dtype=int32)]
print(output[1])


a1 = list()
a1.append(np.array([1,2,3]))
a1.append(np.array([4,5,6]))
a1