# 后端工具类
import os
import cv2 as cv
import joblib
import numpy as np
# 导入该包Pycharm编译时会报错，但不影响运行
import tensorflow.keras.models

# 模型地址
SVM_ENU_MODEL_PATH = 'CPDDModel/model/SVM_ENU.m'
MLP_ENU_MODEL_PATH = 'CPDDModel/model/MLP_ENU.h5'
RES_ENU_MODEL_PATH = 'CPDDModel/model/RES_ENU.h5'
RES_CHN_MODEL_PATH = 'CPDDModel/model/RES_CHN.h5'

# SVM模型所需尺寸
SVM_SIZE = 32

# 设置颜色属性判断时蓝、红、白色的HSV属性阈值，用于像素颜色统计
# 蓝色的HSV属性中的各属性阈值
HSV_MIN_BLUE_H = 90
HSV_MAX_BLUE_H = 130
HSV_MIN_BLUE_SV = 90
HSV_MAX_BLUE_SV = 245

# 红色的HSV属性中的H属性阈值
HSV_MIN1_RED_H = 0
HSV_MAX1_RED_H = 10
HSV_MIN2_RED_H = 156
HSV_MAX2_RED_H = 180

# 白色的HSV属性中的各属性阈值
HSV_MIN_WHITE_H = 0
HSV_MAX_WHITE_H = 180
HSV_MIN_WHITE_S = 0
HSV_MAX_WHITE_S = 30
HSV_MIN_WHITE_V = 221
HSV_MAX_WHITE_V = 255

# 车牌统一的尺寸
PLATE_STD_HEIGHT = 36
PLATE_STD_WEIGHT = 136

# 英文字典
LABEL_ENU_DICT = {
    0: '0', 1: '1', 2: '2', 3: '3', 4: '4', 5: '5', 6: '6', 7: '7', 8: '8', 9: '9',
    10: 'A', 11: 'B', 12: 'C', 13: 'D', 14: 'E', 15: 'F', 16: 'G', 17: 'H',	18: 'J', 19: 'K',
    20: 'L', 21: 'M', 22: 'N', 23: 'P', 24: 'Q', 25: 'R', 26: 'S', 27: 'T', 28: 'U', 29: 'V',
    30: 'W', 31: 'X', 32: 'Y', 33: 'Z'
}

# 中文字典
LABEL_CHN_DICT = {
    0: '川', 1: '鄂', 2: '赣', 3: '甘', 4: '贵', 5: '桂', 6: '黑', 7: '沪', 8: '冀', 9: '津',
    10: '京', 11: '吉', 12: '辽', 13: '鲁', 14: '蒙', 15: '闽', 16: '宁', 17: '青', 18: '琼',
    19: '陕',20: '苏', 21: '晋', 22: '皖', 23: '湘', 24: '新', 25: '豫', 26: '渝', 27: '粤',
    28: '云', 29: '藏',30: '浙', 31: '晋'
}


# 1. 提取车牌区域部分
def preprocess_plate_image_by_sobel(plate_image):
    """
    功能: 使用sobel预处理车辆（含有车辆）图片
    :param plate_image: 车辆（含有车牌）图片
    :return: 预处理后的车牌图片
    """
    # 图片与处理
    # 高斯模糊
    blured_image = cv.GaussianBlur(plate_image, (5, 5), 0)
    # 转成灰度图，此处传入即为灰度图，无需再转灰度
    # gray_image = cv.cvtColor(blured_image, cv.COLOR_BGR2GRAY)
    # gray_image = blured_image
    # 使用Sobel算子，求水平方向一阶导数
    # 使用 cv.CV_16S
    grad_x = cv.Sobel(blured_image, cv.CV_16S, 1, 0, ksize=3)
    # 转成 CV-_8U - 借助 cv.convertScaleAbs()方法
    abs_grad_x = cv.convertScaleAbs(grad_x)
    # 叠加水平和垂直（此处不用）方向，获取 sobel 的输出
    gray_image = cv.addWeighted(abs_grad_x, 1, 0, 0, 0)
    # 二值化操作
    is_success, threshold_image = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)
    # 执行闭操作=>车牌连成矩形区城
    kernel = cv.getStructuringElement(cv.MORPH_RECT, (10, 3))
    morphology_imge = cv.morphologyEx(threshold_image, cv.MORPH_CLOSE, kernel)
    return morphology_imge


def preprocess_plate_image_by_hsv(plate_image):
    """
    功能: 使用hsv提取蓝色区域, 预处理车辆（含有车辆）图片
    :param plate_image:车辆（含有车牌）图片
    :return:预处理后的车牌图片
    """
    # 1. 将一张RGB图像转换为 HSV 图片格式
    hsv_image = cv.cvtColor(plate_image, cv.COLOR_BGR2HSV)
    # 获取h, s, v图片分量，图片h分量的shape
    h_split, s_split, v_split = cv.split(hsv_image)
    rows, cols = h_split.shape
    # 2.遍历图片，找出蓝色的区域
    # 创建全黑背景，== 原始图片大小
    binary_image = np.zeros((rows, cols), dtype=np.uint8)
    # 遍历图片的每一个像素，找到满足条件（hsv找蓝色）的像素点，设置为1 == binary_image中
    for row in np.arange(rows):
        for col in np.arange(cols):
            H = h_split[row, col]
            S = s_split[row, col]
            V = v_split[row, col]
            # 判断像素落在蓝色区域并满足hsv条件
            if ((HSV_MIN_BLUE_H < H < HSV_MAX_BLUE_H) and
                (HSV_MAX_BLUE_SV > S > HSV_MIN_BLUE_SV) and
                (HSV_MAX_BLUE_SV > V > HSV_MIN_BLUE_SV)):
                binary_image[row, col] = 255
    return binary_image


def verify_plate_sizes(contour):
    """
    功能: 判断是否是车牌区域（依据：面积、长宽比）
    :param contour: 某个轮廓-候选车牌区域
    :return:  bool(True/False)
    """
    # 声明常量：长宽比（最小、最大），面积（最小、最大） == 可以微调
    MIN_ASPECT_RATIO = 2.5
    MAX_ASPECT_RATIO = 4.5
    MIN_AREA = 34.0 * 20 * 10
    MAX_AREA = 100.0 * 8 * 100

    # 获取矩形特征描述的 等值线区域，返回：中心点坐标、长和宽、旋转角度 --float
    (center_x, center_y), (w, h), angle = cv.minAreaRect(contour)
    # 获取宽、高=>int
    w = int(w)
    h = int(h)

    # 进行面积判断
    area = w * h
    if area > MAX_AREA or area < MIN_AREA:
        return False

    # 进行长宽比的判断
    aspect_ratio = w / h
    # 判定车牌是否竖排
    if aspect_ratio < 1:
        aspect_ratio = 1.0 / aspect_ratio
        # return False
    # 判定
    if aspect_ratio > MAX_ASPECT_RATIO or aspect_ratio < MIN_ASPECT_RATIO:
        return False
    return True


def verify_pixel_color(plate_image):
    """
    功能: 统计一张图片内颜色为蓝、红、白的像素占比, 以此判断是否是车牌区域
    :param plate_image: 图片
    :return: 是否符合预设标准
    """
    # 1. 将一张RGB 图片转换为 HSV 图片格式
    hsv_image = cv.cvtColor(plate_image, cv.COLOR_BGR2HSV)
    # 获取h、s、v图片分量，图片h分量的shape
    h_split, s_split, v_split = cv.split(hsv_image)
    rows, cols = h_split.shape
    # 2. 遍历图片，找出蓝色区域
    # 各颜色统计数量
    count_white = 0
    count_red = 0
    count_blue = 0
    # 遍历图片的每一个像素, 根据阈值统计像素个数
    for row in np.arange(rows):
        for col in np.arange(cols):
            H = h_split[row, col]
            S = s_split[row, col]
            V = v_split[row, col]
            # 判断像素落在蓝色区域的个数
            if (HSV_MIN_BLUE_H < H < HSV_MAX_BLUE_H) and (
                    HSV_MIN_BLUE_SV < S < HSV_MAX_BLUE_SV) and (
                    HSV_MIN_BLUE_SV < V < HSV_MAX_BLUE_SV):
                count_blue = count_blue + 1
            # 判断像素落在红色区域的个数(与蓝色无交集, 使用elif即可)
            elif HSV_MIN1_RED_H < H < HSV_MAX1_RED_H or HSV_MIN2_RED_H < H < HSV_MAX2_RED_H:
                count_red += 1
            # 判断像素落在白色区域的个数(与蓝色有交集, 另起一个if)
            if(HSV_MIN_WHITE_H < H < HSV_MAX_WHITE_H) and (
                    HSV_MIN_WHITE_S < S < HSV_MAX_WHITE_S) and (
                    HSV_MIN_WHITE_V < V < HSV_MAX_WHITE_V):
                count_white += 1
    # 各颜色占比判断
    # 若杂色占比过多, 放弃该图片
    if (count_red / (row * col)) > 0.04:
        return False
    if (count_white / (row * col)) > 0.4:
        return False
    # 若蓝色占比足够多, 追加到车牌区域的候选区域列表中
    if (count_blue / (row * col)) > 0.27:
        return True
    else:
        return False


def rotate_plate_image(contour, plate_image):
    """
    功能: 车牌旋转矫正(依据：根据长宽判断旋转角度是否需要修正、借助转换|旋转矩阵和原始的图片|扩充图片完成旋转|仿射)
    :param contour: 车牌区域
    :param plate_image: 原始图片
    :return: 完成旋转矫正后的车牌图片
    """
    # 获取车牌区域的正交外接矩形，同时也会返回 长、宽
    # boundingRect 用于获取与等值线框（轮廓框）contour的四个角点正交的矩形
    # 返回 左上的坐标（x, y），宽（w），高（h）
    x, y, w, h = cv.boundingRect(contour)
    # 生成该外接矩阵的图片矩阵:对原始车牌图片的切片提取
    bounding_image = plate_image[y: y+h, x: x+w]
    # 1. 判断并修订旋转角度
    # 获取矩形特征描述的等值线区域，返回：中心点坐标、长和宽、旋转角度
    rect = cv.minAreaRect(contour)
    # 获取整数形式的长\宽
    rect_width, rect_height = np.int0(rect[1])
    # 获取旋转角度|畸变角度
    angle = np.abs(rect[2])
    # 自行调整：1.大小关系。2.角度修订
    if rect_width > rect_height:
        temp = rect_width
        rect_width = rect_height
        rect_height = temp
    # 完成旋转
    # 创建一个放大图片区域，保存旋转之后的结果
    enlarged_width = w * 3 // 2
    enlarged_height = h * 3 // 2
    enlarged_image = np.zeros((enlarged_height, enlarged_width, plate_image.shape[2]), dtype=plate_image.dtype)
    # x,y的放大参数
    x_in_enlarged = (enlarged_width - w) // 2
    y_in_enlarged = (enlarged_height - h) // 2
    # 获取放大图片的居中图片(全0)
    roi_image = enlarged_image[y_in_enlarged:y_in_enlarged + h, x_in_enlarged:x_in_enlarged + w, :]
    # 将旋转前的图片(bounding_image)放置到放大图片的居中位置 == copy
    cv.addWeighted(roi_image, 0, bounding_image, 1, 0, roi_image)
    # 计算旋转中心，就是放大图片的中心
    new_center = (enlarged_width // 2, enlarged_height // 2)
    # 2.开始旋转
    # 计算获取旋转的转换矩阵
    transform_matrix = cv.getRotationMatrix2D(new_center, angle+270, 1.0)
    # 进行|完成旋转：原始图片和旋转转换矩阵的放射计算
    transform_image = cv.warpAffine(enlarged_image, transform_matrix, (enlarged_width, enlarged_height))
    # 获取输出图,截取与最初的等值线框|车牌路径轮廓的长款相同的部分
    output_image = cv.getRectSubPix(transform_image, (rect_height, rect_width), new_center)
    return output_image


def unify_plate_image(plate_image):
    """
    功能: 将车牌图片调整到统一大小
    :param plate_image: 需要统一尺寸的车牌
    :return: 统一之后的统一尺寸的图片
    """
    # resize
    uniformed_image = cv.resize(plate_image, (PLATE_STD_WEIGHT, PLATE_STD_HEIGHT))
    return uniformed_image


def get_candidate_plates_by_sobel(plate_image):
    """
    功能：借助sobel算子完成车牌区域的提取
    :param plate_image: 车牌图片
    :return: 所有可能的车牌候选区域(list类型)
    """
    # 1. 对含有车牌的车辆图片进行预处理(sobel + hsv)
    hsv_image = preprocess_plate_image_by_hsv(plate_image)
    preprocess_image = preprocess_plate_image_by_sobel(hsv_image)
    # 2. 提取所有的等值线|车牌轮廓(可能)的区域
    contours, _ = cv.findContours(preprocess_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # contours_all = cv.findContours(preprocess_image, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE)
    # contours = contours_all[1]
    # 3. 判断并获取所有的车牌区域的候选区域列表
    candidate_plates = []
    # 4. 遍历所有的可能的车牌轮廓|等值线框，筛选出候选区域
    for i in np.arange(len(contours)):
        # candidate_plates.append(contours[i])
        # 逐一获取某一个可能的车牌轮廓区域
        contour = contours[i]
        # 根据面积、长宽比判断是否是候选的车牌区域
        if verify_plate_sizes(contour):
            x, y, w, h = cv.boundingRect(contour)
            if w > h:
                # 完成旋转
                output_image = rotate_plate_image(contour, plate_image)
                # 统一尺寸
                uniformed_image = unify_plate_image(output_image)
                # 像素颜色占比阈值判断
                if verify_pixel_color(uniformed_image):
                    # 追加到车牌候选区域
                    candidate_plates.append(uniformed_image)

    # 返回所有的车辆区域的候选区域列表
    return candidate_plates


# 2. 字符切割部分
def cut_image(image):
    """
    功能: 根据比例，对提取出的车牌进行切割
    :param image: 已提取出的、要进行分割的车牌区域
    :return: 车牌字符(list类型)
    """
    # 获取宽高
    width, height = image.size
    item_width = int(width / 7)
    item_height = height
    item_width_1 = int((width / 70) * 9)
    w = 2 * item_width * 83 / 72
    box_list = []
    # (left, upper, right, lower)
    for i in range(0, 2):
        # 中文字符
        box = (i * item_width, 0, (i + 1) * item_width, item_height)
        box_list.append(box)
    for i in range(2, 7):
        # 英文数字字符
        box = (w + item_width_1 * (i - 2), 0, w + item_width_1 * (i - 1), item_height)
        box_list.append(box)
    image_list = [image.crop(box) for box in box_list]
    return image_list


def char_preprocessing(image_list):
    """
    功能: 对于每个分割出的字符进行预处理
    :param image_list: 字符组
    :return:处理后的字符组
    """
    working_regions = []
    for candidate_char_image in image_list:
        candidate_char_image = cv.cvtColor(np.asarray(candidate_char_image), cv.COLOR_RGB2BGR)
        # 1.图片预处理：灰度+二值化
        gray_image = cv.cvtColor(candidate_char_image, cv.COLOR_BGR2GRAY)
        is_success, binary_image = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)
        # 2向内缩进，去除外边框，横不缩，竖缩5
        # 经验值
        offset_y = 5
        # 切片提取内嵌区域
        offset_region = binary_image[offset_y:-offset_y, :]
        # 生成工作区域
        working_region = offset_region
        working_regions.append(working_region)
    return working_regions


# (测试用)
def save_separated_char(count, working_regions):
    """
    功能: 将分割后经预处理的车牌字符存至同级save_char文件夹下的名为count的文件夹内
    :param count: 文件夹名
    :param working_regions: 分割后经预处理的车牌字符组
    :return: 无
    """
    img_path = os.path.join('save_char', str(count))
    if not os.path.exists(img_path):
        # 如果文件目录不存在则创建目录
        os.makedirs(img_path)
    for i in range(len(working_regions)):
        cv_write(img_path + '/' + str(i) + '.jpg', working_regions[i])


# 3. 字符识别部分
def model_chn_enu():
    """
    功能: 读取模型
    :return: 中文模型、英文及数字模型
    """
    # 读取模型
    model_svm_enu = joblib.load(SVM_ENU_MODEL_PATH)
    model_mlp_enu = tensorflow.keras.models.load_model(MLP_ENU_MODEL_PATH)
    model_res_enu = tensorflow.keras.models.load_model(RES_ENU_MODEL_PATH)
    model_res_chn = tensorflow.keras.models.load_model(RES_CHN_MODEL_PATH)
    return model_res_chn, model_svm_enu, model_mlp_enu, model_res_enu


def normalize_data(data):
    """
    功能: 预处理-标准化
    :param data: 特征矩阵
    :return: 执行标准化后的data
    """
    return (data - data.mean()) / data.max()


def trans_2dims_to_4dims(char):
    """
    功能：将二维的灰度图片(26,19)转换为四维的图片(1,32,32,3)，供模型预测
    :param char: 要转换的二位图片
    :return: 转换后的思维图片
    """
    resized = cv.resize(char, (32, 32))
    # 维度拓展
    expand_0 = np.expand_dims(resized, axis=0)
    expand_1 = np.expand_dims(expand_0, axis=3)
    # 维度堆叠
    stack_0 = np.stack([expand_1, expand_1, expand_1], axis=3)
    # 标准化
    normalized = normalize_data(stack_0)
    return normalized


def mlp_reshape(char):
    """
    功能：将二维的灰度图片(26,19)转换为四维的图片(1,3072)，供模型预测
    :param char: 要转换的二位图片
    :return: 转换后的思维图片
    """
    resized = cv.resize(char, (32, 96))
    resized = resized.reshape(1, 32 * 32 * 3)
    return resized


def get_chn_char_by_int(num):
    """
    功能: 将中文模型预测出的int结果转为中文
    :param num: 中文模型预测出的int结果转
    :return: 对应的中文
    """
    # 中文字典可直接找value值
    return LABEL_CHN_DICT.get(num[0])


# 将英文模型预测出的int结果转为英文
def get_enu_char_by_int(num):
    """
    功能: 将英文模型预测出的int结果转为英文或数字
    :param num: 英文模型预测出的int结果
    :return: 对应的英文或数字
    """
    # 英文字典无直接对应关系，需遍历
    return LABEL_ENU_DICT.get(num[0])


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


def predict_region(chars, model_res_chn, model_svm_enu, model_mlp_enu, model_res_enu):
    """
    功能：识别分割后的单个字符图片组
    :param chars: 分割后的单个字符图片组
    :param model_res_chn: 识别中文的RES模型
    :param model_svm_enu: 识别英文及数字的RES模型
    :param model_mlp_enu: 识别英文及数字的MLP模型
    :param model_res_enu: 识别英文及数字的RES模型
    :return: 识别结果
    """
    # 由于识别中文和英文的模型不同，第一个字符和其余需要分开存储
    result = ''
    # 将图片转换为模型可识别的四维图片并初始化
    # 先识别第一个字符
    normalized_chn = trans_2dims_to_4dims(chars[0])
    # 预测第一个字符
    predict_chn = np.argmax(model_res_chn.predict(normalized_chn), axis=1)
    # 将预测结果的int转换为中文字符
    trans_char = get_chn_char_by_int(predict_chn)
    result += trans_char
    # 对于其余字符
    for char in chars[1:]:
        # 与识别第一个字符同理
        # 按照RES模型所需转换图片
        trans_res = trans_2dims_to_4dims(char)
        # 按照SVM模型所需转换图片
        resized_svm = cv.resize(char, [SVM_SIZE, SVM_SIZE])
        trans_svm = resized_svm.reshape(1, SVM_SIZE * SVM_SIZE)
        # 按照MLP模型所需转换图片
        trans_mlp = mlp_reshape(char)
        # 模型预测
        predict_svm_enu = model_svm_enu.predict(trans_svm)
        predict_mlp_enu = np.argmax(model_mlp_enu.predict(trans_mlp), axis=1)
        predict_res_enu = np.argmax(model_res_enu.predict(trans_res), axis=1)
        # 集成预测结果
        predict_enu = integration(predict_svm_enu, predict_mlp_enu, predict_res_enu)
        # 将识别结果转换为车牌字符
        trans_char = get_enu_char_by_int(predict_enu)
        result += trans_char
    return result


def cv_write(file_path, frame):
    """
    功能：存储为jpg文件
    :param file_path: 要存储为jpg的文件
    :param frame: 存储的地址
    :return: 无
    """
    cv.imencode('.jpg', frame)[1].tofile(file_path)

