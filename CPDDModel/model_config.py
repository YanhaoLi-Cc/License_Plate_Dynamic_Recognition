# 模型常量设置文件

# 定义图像大小
# 图片宽度
IMAGE_WIDTH = 32
# 图片长度
IMAGE_HEIGHT = 32


# 英文与数字类别个数
CLASSIFICATION_COUNT_ENU = 34
# 中文类别个数
CLASSIFICATION_COUNT_CHS = 31


# 英文与数字字典
LABEL_DICT_ENU = {
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9,
    'A': 10, 'B': 11, 'C': 12, 'D': 13, 'E': 14, 'F': 15, 'G': 16, 'H': 17, 'J': 18, 'K': 19,
    'L': 20, 'M': 21, 'N': 22, 'P': 23, 'Q': 24, 'R': 25, 'S': 26, 'T': 27, 'U': 28, 'V': 29,
    'W': 30, 'X': 31, 'Y': 32, 'Z': 33
}


# 中文字典
LABEL_DICT_CHS = {
    'chuan': 0, 'e': 1, 'gan': 2, 'gan1': 3, 'gui': 4, 'gui1': 5, 'hei': 6, 'hu': 7, 'ji': 8, 'jin': 9,
    'jing': 10, 'jl': 11, 'liao': 12, 'lu': 13, 'meng': 14, 'min': 15, 'ning': 16, 'qing': 17, 'qiong': 18, 'shan': 19,
    'su': 20, 'jin1': 21, 'wan': 22, 'xiang': 23, 'xin': 24, 'yu': 25, 'yu1': 26, 'yue': 27, 'yun': 28, 'zang': 29,
    'zhe': 30
}


# 中文训练集地址
TRAIN_DIR_CHS = 'CPDDdata/chinese_train_allbase'
# 中文测试集地址
TEST_DIR_CHS = 'CPDDdata/chinese_test_allbase'


# 英文训练集地址
TRAIN_DIR_ENU = 'CPDDdata/english_train'
# 英文测试集地址
TEST_DIR_ENU = 'CPDDdata/english_test'


# 英文数据预测地址
Predict_DIR_ENU = 'CPDDdata/english_test'
# 中文数据存放地址
Predict_DIR_CHS = 'CPDDdata/chinese_test_allbase'


# ResNet英文模型存放地址
RES_ENU_MODEL_PATH_H5 = './model/RES_ENU.h5'
# ResNet中文模型存放地址
RES_CHS_MODEL_PATH_H5 = './model/RES_SHN.h5'


# MLP英文模型存放地址
MLP_ENU_MODEL_PATH_H5 = 'model/MLP_ENU.h5'
# MLP中文模型存放地址
MLP_CHS_MODEL_PATH_H5 = 'model/MLP_CHS.h5'


# SVM英文模型存放地址
SVM_ENU_PATH = './model/SVM_ENU.m'
# SVM中文模型存放地址
SVM_CHS_PATH = './model/SVM_CHS.m'
