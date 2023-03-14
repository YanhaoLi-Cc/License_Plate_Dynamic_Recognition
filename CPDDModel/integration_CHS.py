# 中文集成识别
import joblib
import numpy as np
import tensorflow as tf
import normalize as nm
import model_config as mc
import load_data as ld
import integration as it


# SVM Model
# 读取数据与标签
SVM_data, SVM_labels = ld.load_path_svm(mc.Predict_DIR_CHS, mc.LABEL_DICT_CHS)
# 标准化数据
SVM_data = nm.normalize_data(SVM_data)
# 调整数据尺寸为SVM模型对应大小
SVM_data = SVM_data.reshape(SVM_data.shape[0], mc.IMAGE_WIDTH * mc.IMAGE_HEIGHT)
# 装载模型
model_SVM = joblib.load(mc.SVM_CHS_PATH)
print("SVM模型装载完毕")
# 开始预测
predicts_SVM = model_SVM.predict(SVM_data)
print('SVM Shape', predicts_SVM.shape)
# 输出预测结果尺寸
errors_SVM = np.count_nonzero(predicts_SVM - SVM_labels)
print('errors_SVM: ', errors_SVM)
# 使用SVM模型的预测正确率accruacy_SVM
accruacy_SVM = (len(predicts_SVM) - errors_SVM) * 1.0 / len(predicts_SVM)
print('accruacy_SVM: ', accruacy_SVM)
inerror_SVM = []
for i in range(len(predicts_SVM)):
    if predicts_SVM[i] != SVM_labels[i]:
        inerror_SVM.append(i)
print('错误标签', inerror_SVM)
print()


# MLP Model
# 读取数据与标签
MLP_data, MLP_labels = ld.load_path(mc.Predict_DIR_CHS, mc.LABEL_DICT_CHS)
# 标准化数据
MLP_data = nm.normalize_data(MLP_data)
# 调整数据尺寸为MLP模型对应大小
MLP_data = MLP_data.reshape(MLP_data.shape[0], mc.IMAGE_WIDTH * mc.IMAGE_HEIGHT * 3)
# 装载模型
model_MLP = tf.keras.models.load_model(mc.MLP_CHS_MODEL_PATH_H5)
print("MLP模型装载完毕")
# 对输出结果进行argmax axis=1
predicts_MLP = np.argmax(model_MLP.predict(MLP_data), axis=1)
# 输出预测结果尺寸
print('MLP Shape', predicts_MLP.shape)
# 使用MLP进行预测的错误条数
errors_MLP = np.count_nonzero(predicts_MLP - MLP_labels)
print('errors_MLP: ', errors_MLP)
# 使用MLP模型的预测正确率accruacy_MLP
accruacy_MLP = (len(predicts_MLP) - errors_MLP) * 1.0 / len(predicts_MLP)
print('accruacy_MLP: ', accruacy_MLP)
inerror_MLP = []
for i in range(len(predicts_MLP)):
    if predicts_MLP[i] != MLP_labels[i]:
        inerror_MLP.append(i)
print('错误标签', inerror_MLP)
print()


# ResNet Model
# 读取数据与标签
RES_data, RES_labels = ld.load_path(mc.Predict_DIR_CHS, mc.LABEL_DICT_CHS)
# 标准化数据
RES_data = nm.normalize_data(RES_data)
# 装载模型
model_RES = tf.keras.models.load_model(mc.RES_CHS_MODEL_PATH_H5)
print("RES模型装载完毕")
# 对输出结果进行argmax axis=1
predicts_RES = np.argmax(model_RES.predict(RES_data), axis=1)
# 输出预测结果尺寸
print('RES Shape', predicts_RES.shape)
# 使用ResNet进行预测的错误条数
errors_RES = np.count_nonzero(predicts_RES - RES_labels)
print('errors_RES: ', errors_RES)
# 使用ResNet模型的预测正确率accruacy_RES
accruacy_RES = (len(predicts_RES) - errors_RES) * 1.0 / len(predicts_RES)
print('accruacy_RES: ', accruacy_RES)
inerror_RES = []
for i in range(len(predicts_RES)):
    if predicts_RES[i] != RES_labels[i]:
        inerror_RES.append(i)
print('错误标签', inerror_RES)
print()



# predict为集成预测结果
predicts = it.integration(predicts_SVM, predicts_MLP, predicts_RES)
# errors为集成预测结果错误条数
errors = np.count_nonzero(predicts - RES_labels)
print('Intergration error: ', errors)
# 输出集成预测准确率
accruacy = (len(predicts) - errors) * 1.0 / len(predicts)
print('Intergration accruacy: ', accruacy)
# 存放集成错误对应标签
inerror = []
for i in range(len(predicts)):
    if predicts[i] != RES_labels[i]:
        inerror.append(i)
print('错误标签', inerror)
