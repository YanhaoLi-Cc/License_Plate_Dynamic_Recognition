# mlp中文训练及测试
import numpy as np
from tensorflow import keras
import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers
import model_config as mc
import load_data as ld
import normalize as nm


def train():
    """
    功能: 模型训练，并保存模型
    :return: 无
    """
    # 读取数据
    train_data, train_labels = ld.load_path(mc.TRAIN_DIR_CHS, mc.LABEL_DICT_CHS)
    # 数据预处理：标准化
    train_data = nm.normalize_data(train_data)
    print('normalized shape: ', train_data.shape)
    # 调整数据尺寸 由四维转为二维
    train_data = train_data.reshape(train_data.shape[0], mc.IMAGE_WIDTH * mc.IMAGE_HEIGHT * 3)
    print('train_data shape: ', train_data.shape)
    # 使用keras.Sequential堆叠构建深度神经网络
    model = keras.Sequential([
        # 全连接层 激活函数为relu 每张图片的输入尺寸为mc.IMAGE_WIDTH * mc.IMAGE_HEIGHT * 3
        layers.Dense(64, activation='relu', input_shape=(mc.IMAGE_WIDTH * mc.IMAGE_HEIGHT * 3,)),
        layers.Dense(64, activation='relu'),
        layers.Dense(64, activation='relu'),
        # 全连接层分类输出 分类个数为CLASSIFICATION_COUNT_CHS 激活函数为softmax
        layers.Dense(mc.CLASSIFICATION_COUNT_CHS, activation='softmax')
    ])
    # 用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
    model.compile(optimizer=keras.optimizers.Adam(),  # 设置优化器为Adam
                  loss=keras.losses.SparseCategoricalCrossentropy(),  # 设置为交叉损失熵函数
                  metrics=['accuracy'])  # 标注网络评价指标 输出正确率
    # 输出模型结构与对应参数个数
    model.summary()
    # 由于中文模型训练会出现过拟合现象，在此每隔3个epoch输出一次测试集loss与accuracy
    for i in range(300):
        # j为每隔3个epoch进入一次测试
        j = 3
        # 如果是第一次训练，直接训练
        if i == 0:
            print("TRAIN")
            model.fit(train_data, train_labels, batch_size=128, epochs=j, verbose=1)
        # 不是第一次训练，先装载模型再开始训练
        else:
            print("TRAIN")
            # 装载模型
            model = tf.keras.models.load_model(mc.MLP_CHS_MODEL_PATH_H5)
            # 开始训练模型
            # 设置batch_size一次训练所抓取的数据样本数量为128
            # 设置训练轮数epochs为3
            # 设置verbose=1 即输出带进度条的输出日志信息
            model.fit(train_data, train_labels, batch_size=128, epochs=j, verbose=1)
        # 输出测试集loss与accuracy
        print(model.evaluate(train_data, train_labels))
        # 保存模型为H5文件
        model.save(mc.MLP_CHS_MODEL_PATH_H5)
        # 如果不是第一次训练并且每隔3个epochs，进入一次测试
        if i % 3 == 0 and i != 0:
            acc = test()
            print('epoch:', i, 'acc:', acc)
            # 设置经验正确率为0.94
            # MLP模型在中文的识别率较低，0.94正确率是该模型所能达到的较好识别结果
            if acc > 0.94:
                return


def test():
    """
    功能: 模型测试，并输出测试结果
    :return: 正确率
    """
    # 读取数据
    test_data, test_labels = ld.load_path(mc.TEST_DIR_CHS, mc.LABEL_DICT_CHS)
    # 数据预处理：标准化
    normalized_test_data = nm.normalize_data(test_data)
    # 调整数据尺寸 由四维转为二维
    normalized_test_data = normalized_test_data.reshape(normalized_test_data.shape[0], mc.IMAGE_WIDTH * mc.IMAGE_HEIGHT * 3)
    # 装载模型
    model = tf.keras.models.load_model(mc.MLP_CHS_MODEL_PATH_H5)
    print('\nTEST')
    # 获取识别损失率loss 正确率acc
    loss, acc = model.evaluate(normalized_test_data, test_labels)
    return acc


def predict(DIR):
    """
    功能: 进行预测，同时使用argmax进行分类，并输出预测结果
    :param DIR: 数据集地址
    :return: 无
    """
    # 读取数据
    predicts_data, _ = ld.load_path(DIR, mc.LABEL_DICT_CHS)
    # 数据预处理
    normalized_predicts_data = nm.normalize_data(predicts_data)
    # 调整数据尺寸 由四维转为二维
    normalized_predicts_data = normalized_predicts_data.reshape(normalized_predicts_data.shape[0], mc.IMAGE_WIDTH * mc.IMAGE_HEIGHT * 3)
    # 装载模型
    model = tf.keras.models.load_model(mc.MLP_CHS_MODEL_PATH_H5)
    # 进行预测，同时使用argmax进行分类 axis设置为1
    predicts = np.argmax(model.predict(normalized_predicts_data), axis=1)
    print(predicts)


train()
test()
predict(mc.TEST_DIR_CHS)