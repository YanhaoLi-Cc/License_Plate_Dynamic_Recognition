# svm中文训练及测试
import numpy as np
from sklearn import svm
import joblib
import normalize as nm
import model_config as mc
import load_data as ld


def train():
    """
    功能: 模型训练，并保存模型
    :return: 无
    """
    print("装载训练数据...")
    # 读取数据 train_data train_labels 分别存放读取后的数据与标签
    train_data, train_labels = ld.load_path_svm(mc.TRAIN_DIR_CHS, mc.LABEL_DICT_CHS)
    # 对train_data进行标准化
    train_data = nm.normalize_data(train_data)
    # 调整train_data数据shape 由三维变为二维
    train_data = train_data.reshape(train_data.shape[0], mc.IMAGE_WIDTH * mc.IMAGE_HEIGHT)
    print("装载%d条数据，每条数据%d个特征" % (train_data.shape))
    print("训练中...")
    # 设置SVM模型为SVC分类模型
    model = svm.SVC(decision_function_shape='ovo')  # ovo代表采用1 vs 1的方式进行多类别分类处理
    # 开始训练
    model.fit(train_data, train_labels)
    print("训练完成，保存模型...")
    # 保存模型
    joblib.dump(model, mc.SVM_CHS_PATH)
    print("模型保存到:", mc.SVM_CHS_PATH)


def test():
    """
    功能: 模型测试，并输出测试结果
    :return: 正确率
    """
    print("装载测试数据...")
    # 读取test_data test_labels数据与标签
    test_data, test_labels = ld.load_path_svm(mc.TEST_DIR_CHS, mc.LABEL_DICT_CHS)
    # 对test_data进行数据标准化
    test_data = nm.normalize_data(test_data)
    # 调整test_data数据尺寸 由三维转为二维
    test_data = test_data.reshape(test_data.shape[0], mc.IMAGE_WIDTH * mc.IMAGE_HEIGHT)
    print("装载%d条数据，每条数据%d个特征" % (test_data.shape))
    print("装载模型...")
    # 加载模型
    model = joblib.load(mc.SVM_CHS_PATH)
    print("模型装载完毕，开始测试...")
    # predicts为预测结果
    predicts = model.predict(test_data)
    # errors为预测测试集的错误数据
    errors = np.count_nonzero(predicts - test_labels)
    # 输出预测正确条数 预测错误条数 正确率
    print("测试完毕，预测正确：%d 条，预测错误:%d 条， 正确率：%f" %
          (len(predicts) - errors, errors, (len(predicts) - errors) / len(predicts)))


train()
test()