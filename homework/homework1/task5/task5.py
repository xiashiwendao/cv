import util
import numpy as np
from pickle import load
from nltk.translate.bleu_score import corpus_bleu
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
import os, sys

def word_for_id(integer, tokenizer):
    """
    将一个整数转换为英文单词
    :param integer: 一个代表英文的整数
    :param tokenizer: 一个预先产生的keras.preprocessing.text.Tokenizer
    :return: 输入整数对应的英文单词
    """
    for word, index in tokenizer.word_index.items():
        if index == integer:
            return word
    return None


def generate_caption(model, tokenizer, photo_feature, max_length = 40):
    """
    根据输入的图像特征产生图像的标题
    :param model: 预先训练好的图像标题生成神经网络模型
    :param tokenizer: 一个预先产生的keras.preprocessing.text.Tokenizer
    :param photo_feature:输入的图像特征, 为VGG16网络修改版产生的特征
    :param max_length: 训练数据中最长的图像标题的长度
    :return: 产生的图像的标题
    """
    in_text = 'startseq'
    for i in range(max_length):
        sequence = tokenizer.texts_to_sequences([in_text])
        sequence = pad_sequences(sequence, maxlen=max_length)
        output = model.predict([photo_feature, sequence])
        best_word_index = np.argmax(output)
        word = word_for_id(best_word_index, tokenizer)
        if word is None:
            break
        in_text = in_text + " " + word
        if word == 'endseq':
            break

    return in_text


def generate_caption_run():
    image_file = 'Flickr_8k.testImages.txt'
    ids = util.load_ids(image_file)
    features = util.load_photo_features('features.pkl', ids)

    nlp_model = load_model('model_19.h5')
    tokenizer = load(open('tokenizer.pkl', 'rb'))

    caption = generate_caption(nlp_model, tokenizer, features['3596131692_91b8a05606'], 40)

    print('+++++++++++ caption is: ', caption, '+++++++++++++++++++')

def evluate_test():
    
    references = [[['1', '2','3','4','5','6','7'],['there', 'is','a','cat','and','a','dog']]]
    candidates = [['there', 'is', 'a', 'cat', 'and','a','pig']]
    score = corpus_bleu(references, candidates, weights=(1,0,0,0))
    print(score)

#evluate_test()

def evaluate_model(model, captions, photo_features, tokenizer, max_length = 40):
    actuals, predicts = list(), list()
    for key, caption_list in captions.items():
        refferences = [d.split() for d in caption_list]
        actuals.append(refferences)
        photo_feature = photo_features[key]
        predict = generate_caption(model, tokenizer, photo_feature, max_length)
        predicts.append(predict.split())

    bleu1 = corpus_bleu(actuals, predicts, weights=(1.0, 0, 0, 0))
    bleu2 = corpus_bleu(actuals, predicts, weights=(0.5, 0.5, 0, 0))
    bleu3 = corpus_bleu(actuals, predicts, weights=(0.3, 0.3, 0.3, 0))
    bleu4 = corpus_bleu(actuals, predicts, weights=(0.25, 0.25, 0.25, 0.25))

    return bleu1, bleu2, bleu3, bleu4

def evaluate_model_run():
    model = load_model('model_19.h5')
    filename = 'Flickr_8k.testImages.txt'
    test = util.load_ids(filename)
    # test play as "index" role, just from description.txt and featute.pkl to 
    # load the special info which define in "index"
    test_caption = util.load_clean_captions('descriptions.txt', test)
    test_features = util.load_photo_features('features.pkl', test)
    tokenizer = load(open('tokenizer.pkl', 'rb'))
    bleu1, bleu2, bleu3, bleu4 = evaluate_model(model, test_caption, test_features, tokenizer)
    print('BLEU-1: %f' % bleu1)
    print('BLEU-2: %f' % bleu2)
    print('BLEU-3: %f' % bleu3)
    print('BLEU-4: %f' % bleu4)



def evaluate_model_my(model, captions, photo_features, tokenizer, max_length = 40):
    """计算训练好的神经网络产生的标题的质量,根据4个BLEU分数来评估

    Args:
        model:　训练好的产生标题的神经网络
        captions: dict, 测试数据集, key为文件名(不带.jpg后缀), value为图像标题list
        photo_features: dict, key为文件名(不带.jpg后缀), value为图像特征
        tokenizer: 英文单词和整数转换的工具keras.preprocessing.text.Tokenizer
        max_length：训练集中的标题的最大长度

    Returns:
        tuple:
            第一个元素为权重为(1.0, 0, 0, 0)的ＢＬＥＵ分数
            第二个元素为权重为(0.5, 0.5, 0, 0)的ＢＬＥＵ分数
            第三个元素为权重为(0.3, 0.3, 0.3, 0)的ＢＬＥＵ分数
            第四个元素为权重为(0.25, 0.25, 0.25, 0.25)的ＢＬＥＵ分数

    """
    score_1 = 0
    score_2 = 0
    score_3 = 0
    score_4 = 0
    for key in photo_features:
        feature = photo_features[key]
        caption_raw = captions[key]
        caption_generated = generate_caption(model, tokenizer, feature)
        score_1 += corpus_bleu([[[caption_raw]]], [[caption_generated]], weights=[1.0, 0, 0, 0])
        score_2 += corpus_bleu([[[caption_raw]]], [[caption_generated]], weights=[0.5, 0.5, 0, 0])
        score_3 += corpus_bleu([[[caption_raw]]], [[caption_generated]], weights=[0.3, 0.3, 0.3, 0])
        score_4 += corpus_bleu([[[caption_raw]]], [[caption_generated]], weights=[0.25, 0.25, 0.25, 0.25])
    total_size = len(photo_features)
    score_1 /= total_size
    score_2 /= total_size
    score_3 /= total_size
    score_4 /= total_size

    return score_1, score_2, score_3, score_4

if __name__ == "__main__":
    # current_path = sys.argv[0]
    # work_path = os.path.abspath(os.path.dirname(current_path)+os.path.sep+".")
    # print("work_path: %s" % work_path)
    # os.chdir(work_path)
    #caption = generate_caption_run()
    #print('caption is: %s' % caption)
    generate_caption_run()
    evaluate_model_run()