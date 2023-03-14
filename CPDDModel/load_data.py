# 加载数据
import numpy as np
import cv2 as cv
import os
import model_config as mc


def load_path_svm(dir_path, label_dict):
    """
    功能: SVM模型读取数据，需要将颜色通道数变为1
    :param dir_path: 数据集目录
    :param label_dict: 图片标签列表
    :return: 样本数据特征矩阵、标签向量
    """
    # 分别存放图片与对应标签
    data = []
    labels = []
    # 获取数据集目录下的所有的子目录，并逐一遍历
    for item in os.listdir(dir_path):
        # 获取每一个具体样本类型的 os 的路径形式
        # item_path为不同类别样本文件夹路径
        item_path = os.path.join(dir_path, item)
        # 判断只有目录，才进行下一级目录的遍历
        if os.path.isdir(item_path):
            # 到了每一个样本目录，遍历其下的每个样本文件--图片
            for subitem in os.listdir(item_path):
                # subitem_path为同一类别内不同图片的文件路径
                subitem_path = os.path.join(item_path, subitem)
                # 读取图片
                image = cv.imread(subitem_path)
                # 转为灰度图
                image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
                # 调整图片大小为 mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT
                resized_image = cv.resize(image, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
                # 将图片、标签分别存入对应列表
                data.append(resized_image)
                labels.append(label_dict[item])
    # 分别赋值 样本数据特征 样本数据标签
    features = np.array(data, dtype=np.float32)
    labels = np.array(labels)
    # 返回特征矩阵、标签向量
    return features, labels


def load_path(dir_path, LABEL_DICT):
    """
    功能: MLP与ResNet模型读取数据
    :param dir_path: 数据集目录
    :param LABEL_DICT: 图片标签列表
    :return: 样本数据特征矩阵、标签向量
    """
    # 分别存放图片与对应标签
    data = []
    labels = []
    # 获取数据集目录下的所有的子目录，并逐一遍历
    for item in os.listdir(dir_path):
        # 获取每一个具体样本类型的 os 的路径形式
        # item_path为不同类别样本文件夹路径
        item_path = os.path.join(dir_path, item)
        # 判断只有目录，才进行下一级目录的遍历
        if os.path.isdir(item_path):
            # 到了每一个样本目录，遍历其下的每个样本文件--图片
            for subitem in os.listdir(item_path):
                # subitem_path为同一类别内不同图片的文件路径
                subitem_path = os.path.join(item_path, subitem)
                # 读取图片
                image = cv.imread(subitem_path)
                # 调整图片大小为 mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT
                resized_image = cv.resize(image, (mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT))
                # 将图片、标签分别存入对应列表
                data.append(resized_image)
                labels.append(LABEL_DICT[item])
    # 分别赋值 样本数据特征 样本数据标签
    features = np.array(data)
    labels = np.array(labels)
    # 返回特征矩阵、标签向量
    return features, labels

