# 对数据进行预处理：标准化
def normalize_data(data):
    """
    功能: 对数据进行预处理：标准化
    :param data: 要进行预处理的数据
    :return: 预处理后数据
    """
    return (data - data.mean()) / data.max()
