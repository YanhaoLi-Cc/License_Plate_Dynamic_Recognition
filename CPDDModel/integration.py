# 集成预测函数
def integration(predicts_SVM, predicts_MLP, predicts_RES):
    """
    功能: 根据SVM, MLP, RES三个模型的预测结果进行集成预测
    :param predicts_SVM: SVM模型预测结果
    :param predicts_MLP: MLP模型预测结果
    :param predicts_RES: RES模型预测结果
    :return: 集成预测结果
    """
    # 存放最后预测结果
    predict = []
    for i in range(predicts_RES.shape[0]):
        # svm mlp res分别对应预测结果
        svm = predicts_SVM[i]
        mlp = predicts_MLP[i]
        res = predicts_RES[i]
        # 三个预测结果都相同
        if svm == mlp and svm == res:
            predict.append(svm)
            continue
        # 三个预测结果中两个相同
        if svm == mlp and svm != res:
            predict.append(svm)
            continue
        if svm == res and svm != mlp:
            predict.append(svm)
            continue
        if res == mlp and res != svm:
            predict.append(svm)
            continue
        # 三个预测结果都不相同时
        # 取ResNet的预测结果作为最后结果
        if svm != res and res != mlp and svm != mlp:
            predict.append(res)
            continue
    return predict