# 实现图片识别
import os
import cv2 as cv
import numpy as np
from PIL import Image
import util


def get_plate_single(plate_image):
    """
    功能: 对单个图片提取车牌
    :param plate_image:  图片文件
    :return: 车牌区域
    """
    # 遍历获取所有的车牌图片，逐一(×)
    # 提取每个候选区域列表的第一张图即可
    candidate_plates = util.get_candidate_plates_by_sobel(plate_image)
    if len(candidate_plates):
        return candidate_plates[0]


def cut_image(image):
    """
    功能: 实现对车牌字符的分割
    :param image: 提取的车牌图片
    :return: 车牌字符组(list类型)
    """
    image_list = util.cut_image(image)
    working_regions = util.char_preprocessing(image_list)
    return working_regions


def main(file_path):
    """
    功能: 进行单张图片识别
    :param file_path: 图片文件地址
    :return: 识别结果
    """
    # 路径可包含中文
    plate_file_path = file_path
    # 读取文件
    plate_image = cv.imdecode(np.fromfile(os.path.join(plate_file_path), dtype=np.uint8), -1)
    # 获取车牌区域
    candidate_plate = get_plate_single(plate_image)
    try:
        candidate_image = Image.fromarray(cv.cvtColor(candidate_plate, cv.COLOR_BGR2RGB))
        # 字符分割
        chars = cut_image(candidate_image)
        # 获取模型
        model_res_chn, model_svm_enu, model_mlp_enu, model_res_enu = util.model_chn_enu()
        # 模型预测
        result = util.predict_region(chars, model_res_chn, model_svm_enu, model_mlp_enu, model_res_enu)
        return result
    except cv.error as e:
        return '未检测到车牌'
