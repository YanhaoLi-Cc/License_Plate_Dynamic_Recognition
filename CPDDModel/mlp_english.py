# mlp英文训练及测试
import numpy as np
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import load_data as ld
import model_config as mc
import normalize as nm


def train():
    """
    功能: 模型训练，并保存模型
    :return: 无
    """
    # 读取数据
    train_data, train_labels = ld.load_path(mc.TRAIN_DIR_ENU, mc.LABEL_DICT_ENU)
    # 数据预处理：标准化
    train_data = nm.normalize_data(train_data)
    print('normalized shape: ', train_data.shape)
    # 调整数据尺寸 由四维转变为二维
    train_data = train_data.reshape(train_data.shape[0], mc.IMAGE_WIDTH * mc.IMAGE_HEIGHT * 3)
    print('train data shape: ', train_data.shape)
    # 使用keras.Sequential堆叠构建深度神经网络
    model = keras.Sequential([
        # 全连接层 激活函数为relu 每张图片的输入尺寸为mc.IMAGE_WIDTH * mc.IMAGE_HEIGHT * 3
        layers.Dense(64, activation='relu', input_shape=(mc.IMAGE_WIDTH * mc.IMAGE_HEIGHT * 3,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        # 全连接层分类输出 分类个数为CLASSIFICATION_COUNT_ENU 激活函数为softmax
        layers.Dense(mc.CLASSIFICATION_COUNT_ENU, activation='softmax')
    ])
    # 用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
    model.compile(optimizer=keras.optimizers.Adam(),  # 设置优化器为Adam
                  loss=keras.losses.SparseCategoricalCrossentropy(),  # 设置为交叉损失熵函数
                  metrics=['accuracy'])  # 标注网络评价指标 输出正确率
    # 输出模型结构
    model.summary()
    # 开始训练模型
    # 设置batch_size一次训练所抓取的数据样本数量为128
    # 设置训练轮数epochs为30
    # 设置verbose=1 即输出带进度条的输出日志信息
    model.fit(train_data, train_labels, batch_size=128, epochs=30, verbose=1)
    # 输出训练集损失率loss与正确率accuracy
    print(model.evaluate(train_data, train_labels))
    # 保存模型为H5文件
    model.save(mc.MLP_ENU_MODEL_PATH_H5)


def test():
    """
    功能: 模型测试，并输出测试结果
    :return: 正确率
    """
    # 读取数据
    test_data, test_labels = ld.load_path(mc.TEST_DIR_ENU, mc.LABEL_DICT_ENU)
    # 数据预处理：标准化
    normalized_test_data = nm.normalize_data(test_data)
    print(normalized_test_data.shape)
    # 调整数据尺寸大小 由四维转变为二维
    normalized_test_data = normalized_test_data.reshape(normalized_test_data.shape[0],
                                                        mc.IMAGE_WIDTH * mc.IMAGE_HEIGHT * 3)
    # 装载模型
    model = tf.keras.models.load_model(mc.MLP_ENU_MODEL_PATH_H5)
    print('\nTEST')
    # 输出测试集损失率loss与正确率accuracy
    print(model.evaluate(normalized_test_data, test_labels))


def predict(DIR):
    """
    功能: 进行预测，同时使用argmax进行分类，并输出预测结果
    :param DIR: 数据集地址
    :return: 无
    """
    # 读取需要预测的数据
    predicts_data, _ = ld.load_path(DIR, mc.LABEL_DICT_ENU)
    # 数据预处理：标准化
    normalized_predicts_data = nm.normalize_data(predicts_data)
    # 调整数据尺寸 由4维转为2维
    normalized_predicts_data = normalized_predicts_data.reshape(normalized_predicts_data.shape[0],
                                                                mc.IMAGE_WIDTH * mc.IMAGE_HEIGHT * 3)
    # 装载模型
    model = tf.keras.models.load_model(mc.MLP_ENU_MODEL_PATH_H5)
    # 进行预测，同时使用argmax进行分类 axis设置为1
    predicts = np.argmax(model.predict(normalized_predicts_data), axis=1)
    print(predicts)


train()
test()
predict(mc.TEST_DIR_ENU)
