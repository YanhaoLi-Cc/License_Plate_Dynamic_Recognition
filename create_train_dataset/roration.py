# 图片旋转，增加数据集数量
import os
import pathlib
import cv2 as cv
import numpy as np
from PIL import Image

IMG_PATH = "E:/software/zhongruan/zhongruan_class/Workspace/ZR_project/4_char_division/char_from_allbase_3"


def cv_imread(file_path):
    """
    功能: 读取文件
    :param file_path: 文件地址
    :return: 图片文件
    """
    img_mat=cv.imdecode(np.fromfile(file_path,dtype=np.uint8),-1)
    return img_mat


def read_path(file_pathname):
    """
    功能: 图片读取并旋转
    :param file_pathname:图片所在的路径
    :return: 无
    """
    file_pathname = pathlib.Path(file_pathname)
    k = -1
    for jpg_path in file_pathname.glob('*/*.jpg'):
        k = k+1
        print(jpg_path)
        # tmp_path = os.path.basename(jpg_path)
        # print(tmp_path)
        img = Image.open(jpg_path)
        im_rotate = img.rotate(2)
        str_jpg_path = str(jpg_path)
        filename = str_jpg_path.split("\\")[2]
        # print(filename)
        img_path = IMG_PATH + "/" + filename
        if not os.path.exists(img_path):
            # 如果文件目录不存在则创建目录
            os.makedirs(img_path)
        im_rotate.save(IMG_PATH + "/" + filename + "/" + "rorate3_"+str(k)+".jpg")


if __name__ == '__main__':
    file_pathname = 'data'
    read_path(file_pathname)
