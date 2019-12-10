from pickle import load


def load_doc(filename):
    """读取文本文件为string

    Args:
        filename: 文本文件

    Returns:
        string, 文本文件的内容
    """
    # open the file as read only
    file = open(filename, 'r')
    # read all text
    text = file.read()
    # close the file
    file.close()
    return text


def to_list(captions):
    """将一个字典(key为文件名, value为图像标题list)转换为图像标题list

    Args:
        captions: 一个字典, key为文件名, value为图像标题list

    Returns:
        图像标题list

    """
    all_captions = list()
    for key in captions.keys():
        [all_captions.append(d) for d in captions[key]]
    return all_captions


def get_max_length(captions):
    """从标题字典计算图像标题里面最长的标题的长度

    Args:
        captions: 一个dict, key为文件名(不带.jpg后缀), value为图像标题list

    Returns:
        最长标题的长度

    """
    lines = to_list(captions)
    return max(len(d.split()) for d in lines)


def load_image_names(filename):
    """从文本文件加载图像名set

    Args:
        filename: 文本文件,每一行都包含一个图像文件名（包含.jpg文件后缀）

    Returns:get_max_length
        set, 文件名，去除了.jpg后缀
    """

    doc = load_doc(filename)
    dataset = list()
    # process line by line
    for line in doc.split('\n'):
        # skip empty lines
        if len(line) < 1:
            continue
        # get the image identifier
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)


def load_clean_captions(filename, dataset):
    """为图像标题首尾分别加上'startseq ' 和 ' endseq', 作为自动标题生成的起始和终止

    Args:
        filename: 文本文件,每一行由图像名,和图像标题构成, 图像的标题已经进行了清洗
        dataset: 图像名list

    Returns:
        dict, key为图像名, value为添加了＇startseq'和＇endseq'的标题list
    """

    # load document
    doc = load_doc(filename)
    caption_dict = dict()
    for line in doc.split('\n'):
        # split line by white space
        tokens = line.split()
        # split id from description
        image_id, image_desc = tokens[0], tokens[1:]
        # skip images not in the set
        if image_id in dataset:
            # create list
            if image_id not in caption_dict:
                caption_dict[image_id] = list()
            # wrap description in tokens
            desc = 'startseq ' + ' '.join(image_desc) + ' endseq'
            # store
            caption_dict[image_id].append(desc)
    return caption_dict


def load_photo_features(filename, dataset):
    """从图像特征文件中加载给定图像名list对应的图像特征

    Args:
        filename: 包含图像特征的文件名, 文件加载以后是一个字典,
                    key为'Flicker8k_Dataset/' + 文件名,
                    value为文件名对应的图表的特征
        dataset: 图像文件名list

    Returns:
        图像特征字典, key为文件名,
                    value为文件名对应的图表的特征

    """
    # load all features
    all_features = load(open(filename, 'rb'))
    # filter features
    features = {k: all_features[k] for k in dataset}
    return features
