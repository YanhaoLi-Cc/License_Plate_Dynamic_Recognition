# res中文训练及测试
from tensorflow.keras import layers
import tensorflow.keras.models
import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
from keras.models import Model
from tensorflow.python.ops.init_ops_v2 import glorot_uniform
import keras.backend as K
import normalize as nm
import load_data as ld
import model_config as mc


# 设置默认图像的维度顺序 颜色通道数在第三维
K.set_image_data_format('channels_last')
# 设置训练状态为1 train
K.set_learning_phase(1)


def identity_block(X, f, filters, stage, block):
    """
    ResNet基本网络结构之一
    Identity Block输入维度和输出维度相同，可以串联，用于加深网络

    Arguments:
    X -- tensor的输入尺寸 (m, n_H_prev, n_W_prev, n_C_prev)
    f -- 指定主路径的中间conv的形状
    filters -- 定义主路径的过滤器数量
    stage -- 用于命名图层，具体取决于它们在网络中的位置
    block -- 根据位置给block命名

    Returns:
    X -- identity block层输出 尺寸为(n_H, n_W, n_C)
    """

    # 定义名称
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # 检索过滤器
    F1, F2, F3 = filters

    # 保存输入值。稍后需要用到它才能添加回主路径。
    X_shortcut = X

    # 主路径的第一组件
    # Conv2D实现了输入张量与设定卷积核的卷积操作
    # 设置卷积核filters为F1 卷积操作后结果输出的通道数, 相当于图像卷积后输出的图像通道数。
    # 卷积核大小为(1, 1) 步长为(1, 1) 图像边缘填充为valid 直接舍弃卷积后的剩余像素
    # glorot_uniform为均匀初始化器 参数从[-limit, limit]的均匀分布产生 seed：随机数种子为0
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name=conv_name_base + "2a", kernel_initializer=glorot_uniform(seed=0))(X)
    # BN层 进行数据归一化 最重要的作用是让加速网络的收敛速度
    # axis: 应规范化的轴(要素轴）由于data_format="channels_last" 因此在BatchNormalization中设置axis=3
    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    # 设置激活函数为Relu
    X = Activation("relu")(X)

    # 主路径的第二组件
    # 设置卷积核filters为F1 卷积操作后结果输出的通道数, 相当于图像卷积后输出的图像通道数。
    # 卷积核大小为(f, f) 步长为(1, 1) 图像边缘填充为same 使卷积输出与输入保持相同的shape
    # glorot_uniform为均匀初始化器 参数从[-limit, limit]的均匀分布产生 seed：随机数种子为0
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",
               name=conv_name_base + "2b", kernel_initializer=glorot_uniform(seed=0))(X)
    # BN层 进行数据归一化 最重要的作用是让加速网络的收敛速度
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    # 设置激活函数为Relu
    X = Activation("relu")(X)

    # 主路径的第三组件：将快捷键值添加到主路径，并通过 RELU 激活传递
    # 设置卷积核filters为F3 卷积操作后结果输出的通道数
    # 卷积核大小为(1, 1) 步长为(1, 1) 图像边缘填充为valid直接舍弃卷积后的剩余像素
    # glorot_uniform为均匀初始化器 参数从[-limit, limit]的均匀分布产生 seed：随机数种子为0
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name=conv_name_base + "2c", kernel_initializer=glorot_uniform(seed=0))(X)
    # BN层 进行数据归一化
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)
    # 原先的X与经过Identity Block的X合并主路径 进入relu激活函数
    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Conv Block输入和输出的维度是不一样的，所以不能连续串联，它的作用是改变网络的维度

    Arguments:
    X -- tensor的输入尺寸 (m, n_H_prev, n_W_prev, n_C_prev)
    f -- 指定主路径的中间conv的形状
    filters -- 定义义主路径的过滤器数量
    stage -- 用于命名图层，具体取决于它们在网络中的位置
    block -- 根据位置给block命名
    s -- 指定要使用的步长

    Returns:
    X -- convolutional block层输出 输出尺寸为(n_H, n_W, n_C)
    """

    # 定义名称
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # 检索过滤器
    F1, F2, F3 = filters

    # 保存输入值。稍后需要用到它才能添加回主路径。
    X_shortcut = X

    # convolutional_block左层结构
    # 主路径的第一组件
    # Conv2D实现了输入张量与设定卷积核的卷积操作
    # 设置卷积核filters为F1 卷积操作后结果输出的通道数, 相当于图像卷积后输出的图像通道数。
    # 卷积核大小为(1, 1) 步长为(s, s) 图像边缘填充为valid 直接舍弃卷积后的剩余像素
    # glorot_uniform为均匀初始化器 参数从[-limit, limit]的均匀分布产生 seed：随机数种子为0
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    # BN层 进行数据归一化
    # axis: 应规范化的轴(要素轴）由于data_format=“channels_last 因此在BatchNormalization中设置axis=3
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    # 设置激活函数为Relu
    X = Activation('relu')(X)

    # 主路径的第二组件
    # 设置卷积核filters为F2 卷积操作后结果输出的通道数, 相当于图像卷积后输出的图像通道数。
    # 卷积核大小为(f, f) 步长为(1, 1) 图像边缘填充为same 使卷积输出与输入保持相同的shape
    # glorot_uniform为均匀初始化器
    X = Conv2D(F2, (f, f), strides=(1, 1), name=conv_name_base + '2b', padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    # BN层 进行数据归一化
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    # 设置激活函数为Relu
    X = Activation('relu')(X)

    # 主路径第三组件
    # 设置卷积核filters为F3 卷积操作后结果输出的通道数, 相当于图像卷积后输出的图像通道数。
    # 卷积核大小为(1, 1) 步长为(1, 1) 图像边缘填充为valid 直接舍弃卷积后的剩余像素
    # glorot_uniform为均匀初始化器 参数从[-limit, limit]的均匀分布产生 seed：随机数种子为0
    X = Conv2D(F3, (1, 1), strides=(1, 1), name=conv_name_base + '2c', padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    # BN层 进行数据归一化 最重要的作用是让加速网络的收敛速度
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    # convolutional_block右层结构
    # Conv2D实现了输入张量与设定卷积核的卷积操作
    # 设置卷积核filters为F1 卷积操作后结果输出的通道数, 相当于图像卷积后输出的图像通道数。
    # 卷积核大小为(1, 1) 步长为(s, s) 图像边缘填充为valid 直接舍弃卷积后的剩余像素
    # glorot_uniform为均匀初始化器 seed：随机数种子为0
    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1', padding='valid',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    # BN层 进行数据归一化
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # convolutional_block内左右层合并 再经过Relu后输出
    X = layers.add([X, X_shortcut])
    X = Activation('relu')(X)

    return X


# 3. 训练
def train():
    """
    功能: 模型训练，并保存模型
    :return: 无
    """
    # 加载训练数据：
    train_data, train_labels = ld.load_path(mc.TRAIN_DIR_CHS, mc.LABEL_DICT_CHS)
    # 数据预处理
    normalized_data = nm.normalize_data(train_data)

    # 模型加载
    # 定义输入tensor的尺寸大小为(mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT, 3)
    X_input = Input((mc.IMAGE_WIDTH, mc.IMAGE_HEIGHT, 3))

    # 进行补零操作
    # padding = (3, 3) 以上下、左右对称的方式填充0
    # 表示上下各填充三行0，即：行数加6；左右各填充三列0，即：列数加6
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    # Conv2D实现了输入张量与设定卷积核的卷积操作
    # 设置卷积核filters为64
    # 卷积核大小为(7, 7) 步长为(2, 2)
    # glorot_uniform为均匀初始化器 seed：随机数种子为0
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name="conv",
               kernel_initializer=glorot_uniform(seed=0))(X)
    # BN层 进行数据归一化
    # axis: 应规范化的轴(要素轴）由于data_format="channels_last" 因此在BatchNormalization中设置axis=3
    X = BatchNormalization(axis=3, name="bn_conv1")(X)
    # 设置激活函数为Relu
    X = Activation("relu")(X)
    # 设置池化层为MaxPooling2D最大池化
    # 池化核大小为(3, 3) 池化核步长为(2, 2)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # Stage 2
    # 进入convolutional_block
    # 设置convolutional_block第二组件的卷积核大小为(3,3) 卷积后输出的通道数为[64, 64, 256] Stage2
    X = convolutional_block(X, f=3, filters=[64, 64, 256], stage=2, block="a", s=1)
    # 进入两次identity_block
    # 设置identity_block第二组件的卷积核大小为(3,3) 卷积后输出的通道数为[64, 64, 256] Stage2
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="b")
    X = identity_block(X, f=3, filters=[64, 64, 256], stage=2, block="c")

    # Stage 3
    # 进入convolutional_block
    # 设置convolutional_block第二组件的卷积核大小为(3,3) 卷积后输出的通道数为[128, 128, 512] Stage3
    X = convolutional_block(X, f=3, filters=[128, 128, 512], stage=3, block="a", s=1)
    # 进入三次identity_block
    # 设置identity_block第二组件的卷积核大小为(3,3) 卷积后输出的通道数为[128, 128, 512] Stage3
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="b")
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="c")
    X = identity_block(X, f=3, filters=[128, 128, 512], stage=3, block="d")

    # Stage 4
    # 进入convolutional_block
    # 设置convolutional_block第二组件的卷积核大小为(3,3) 卷积后输出的通道数为[256, 256, 1024] Stage4
    X = convolutional_block(X, f=3, filters=[256, 256, 1024], stage=4, block="a", s=2)
    # 进入五次identity_block
    # 设置identity_block第二组件的卷积核大小为(3,3) 卷积后输出的通道数为[256, 256, 1024] Stage4
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="b")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="c")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="d")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="e")
    X = identity_block(X, f=3, filters=[256, 256, 1024], stage=4, block="f")

    # Stage 5
    # 进入convolutional_block
    # 设置convolutional_block第二组件的卷积核大小为(3,3) 卷积后输出的通道数为[512, 512, 2048] Stage5
    X = convolutional_block(X, f=3, filters=[512, 512, 2048], stage=5, block="a", s=2)
    # 进入两次identity_block
    # 设置identity_block第二组件的卷积核大小为(3,3) 卷积后输出的通道数为[512, 512, 2048] Stage5
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="b")
    X = identity_block(X, f=3, filters=[512, 512, 2048], stage=5, block="c")

    # 进入平均池化层AveragePooling2D 池化层大小为(2, 2)
    # padding="same" 图像边缘填充为same 使卷积输出与输入保持相同的shape
    X = AveragePooling2D(pool_size=(2, 2), padding="same")(X)

    # 输出层
    # 展平X
    X = Flatten()(X)
    # 全连接层 分类个数为mc.CLASSIFICATION_COUNT_ENU
    # 激活函数为softmax glorot_uniform为均匀初始化器
    X = Dense(mc.CLASSIFICATION_COUNT_CHS, activation="softmax", name="fc" + str(34), kernel_initializer=glorot_uniform(seed=0))(X)

    # 创建模型ResNet50
    model = Model(inputs=X_input, outputs=X, name="ResNet50")

    # 输出模型结构与对应参数个数
    model.summary()

    # 用于在配置训练方法时，告知训练时用的优化器、损失函数和准确率评测标准
    model.compile(optimizer='adam',  # 设置优化器为Adam
                  loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                  # 设置为交叉损失熵函数 from_logits=True需要经过激活函数的处理
                  metrics=['accuracy'])  # 标注网络评价指标 输出正确率

    # 设置总迭代次数epochs为3*67=201
    for i in range(67):
        # 每隔3个epoch进行一次测试，防止过拟合
        j = 3
        if i == 0:
            # 第一次直接训练 设定batch_size=128 3个epochs为一个周期
            model.fit(normalized_data, train_labels, epochs=j, batch_size=128)
        else:
            # 不是第一次训练 先导入模型后再训练 设定batch_size=34 3个epochs为一个周期
            model = tensorflow.keras.models.load_model(mc.RES_CHS_MODEL_PATH_H5)
            model.fit(normalized_data, train_labels, epochs=j, batch_size=34)
        # 保存ResNet模型为H5文件
        model.save(mc.RES_CHS_MODEL_PATH_H5)

        # 隔3个epochs 开始测试与评估
        # 导入测试数据
        test_data, test_labels = ld.load_path(mc.TEST_DIR_CHS, mc.LABEL_DICT_CHS)
        # 对测试数据进行标准化处理
        normalized_data_1 = nm.normalize_data(test_data)
        # 加载模型
        model2 = tensorflow.keras.models.load_model(mc.RES_CHS_MODEL_PATH_H5)
        # 进行预测
        predicts = np.argmax(model2.predict(normalized_data_1), axis=1)
        # 输出错误条数
        errors = np.count_nonzero(predicts - test_labels)
        print(errors)
        # 输出正确率
        accruacy = (len(predicts) - errors) * 1.0 / len(predicts)
        print(accruacy)
        # 输出此时所经过的epochs
        print(j * (i + 1))
        # 如果正确率到达0.995 停止训练
        if accruacy > 0.985:
            return


train()