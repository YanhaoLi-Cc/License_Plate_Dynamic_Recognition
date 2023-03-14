# 图片加噪点，增加数据集数量
import os
import pathlib
import random
import cv2 as cv
import numpy as np
from PIL import Image

# 图片存储文件夹
IMG_PATH = "E:\\software\\zhongruan\\zhongruan_class\\Project_plate\\char_from_20000_3\\char_from_20000_3"


def cv_imread(file_path ):
    """
    功能: 读取文件
    :param file_path: 文件地址
    :return: 图片文件
    """
    img_mat = cv.imdecode(np.fromfile(file_path, dtype=np.uint8), -1)
    return img_mat


def cv_imwrite(file_path, frame):
    """
    功能: 存储文件
    :param file_path:
    :param frame:
    :return:
    """
    cv.imencode('.jpg', frame)[1].tofile(file_path)


def read_path(file_pathname):                       # 函数的输入是图片所在的路径
    """
    功能: 图片读取并加噪点，存储加完噪点的图片
    :param file_pathname:
    :return: 无
    """
    file_pathname = pathlib.Path(file_pathname)
    k = -1
    for jpg_path in file_pathname.glob('*/*.jpg'):
        k=k+1
        print(jpg_path)
        str_jpg_path = str(jpg_path)
        img = cv_imread(str_jpg_path)  # 读取图片
        # im_noise=sp_noise(img,0.05)
        im_noise = gaussian_noise(img, 0, 0.0001)
        filename = str_jpg_path.split("\\")[-2]
        # print(filename)
        img_path = IMG_PATH + "/" + filename
        if not os.path.exists(img_path):
            # 如果文件目录不存在则创建目录
            os.makedirs(img_path)
        cv_imwrite(IMG_PATH + "/" + filename + "/" + "noise_"+str(k)+".jpg", im_noise)


def sp_noise(noise_img, proportion):
    """
    功能: 添加椒盐噪声
    :param noise_img: 要加噪音的图片
    :param proportion: 加入噪声的量，可根据需要自行调整
    :return: 加噪音后的图片
    """
    # 获取高度宽度像素值
    height, width = noise_img.shape[0], noise_img.shape[1]
    # 一个准备加入多少噪声小点
    num = int(height * width * proportion)
    for i in range(num):
        w = random.randint(0, width - 1)
        h = random.randint(0, height - 1)
        if random.randint(0, 1) == 0:
            noise_img[h, w] = 0
        else:
            noise_img[h, w] = 255
    return noise_img


def gaussian_noise(img, mean, sigma):
    """
    功能: 将产生的高斯噪声加到图片上
    :param img: 原图
    :param mean: 均值
    :param sigma: 标准差
    :return: 噪声处理后的图片
    """
    # 将图片灰度标准化
    img = img / 255
    # 产生高斯 noise
    noise = np.random.normal(mean, sigma, img.shape)
    # 将噪声和图片叠加
    gaussian_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    gaussian_out = np.clip(gaussian_out, 0, 1)
    # 将图片灰度范围的恢复为 0-255
    gaussian_out = np.uint8(gaussian_out*255)
    # 将噪声范围搞为 0-255
    # noise = np.uint8(noise*255)
    # 这里也会返回噪声，注意返回值
    return gaussian_out


if __name__ == '__main__':
    file_pathname = 'data'
    read_path(file_pathname)
