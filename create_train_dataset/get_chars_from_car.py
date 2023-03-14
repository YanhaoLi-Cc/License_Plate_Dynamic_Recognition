# 将car_photo文件夹中的所有图片提取车牌区域并分割字符，生成训练集存至char_from_plate文件夹
"""
注: 本代码文件仅为演示，拆分结果为一整个数据集且中文汉字未用拼音表达，
    实际模型训练时，需要人为将中文文件夹名改为拼音，并拆解整个数据集为训练集和测试集
"""
import numpy as np
import cv2 as cv
import os
from PIL import Image

"""
CCPD数据集没有专门的标注文件，每张图像的文件名就是该图像对应的数据标注。例如名为
    ”3061158854166666665-97_100-159&434_586&578-558&578_173&523_159&434_586&474-0_0_3_24_33_32_28_30-64-233.jpg“
的图片，将图片名由分割符’-'分为多个部分，每部分有下列含义:
    3061158854166666665: 区域编号(无较大意义)
    97_100: 车牌的两个倾斜角度：水平倾斜角和垂直倾斜角。此处为水平倾斜97度, 竖直倾斜100度。
        (水平倾斜度是车牌与水平线之间的夹角。二维旋转后，垂直倾斜角为车牌左边界线与水平线的夹角)
    159&434_586&578: 边界框左上角和右下角坐标: 左上(159, 434), 右下(586, 578)
    558&578_173&523_159&434_586&474: 车牌四个顶点坐标(右下角开始顺时针排列):
        此处为: 右下(558, 578), 左下(173, 523), 左上(159, 434), 右上(586, 474)
    0_0_3_24_33_32_28_30: 车牌号码(第一位为省份缩写), 在CCPD2019中这个参数为7位, CCPD2020中为8位, 有对应的关系表
    64 为亮度, 数值越大车牌越亮
    233 为模糊度, 数值越小车牌越模糊

所以，可根据CCPD中文件名获取图片标签以构建训练集。(预测功能仍需使用opencv实现车牌字符识别)
"""

# 英文及数字列表
WORDS_LIST = [
    "A", "B", "C", "D", "E",
    "F", "G", "H", "J", "K",
    "L", "M", "N", "P", "Q",
    "R", "S", "T", "U", "V",
    "W", "X", "Y", "Z", "0",
    "1", "2", "3", "4", "5",
    "6", "7", "8", "9"
]

# 中文列表
CON_LIST = [
    "皖", "沪", "津", "渝", "冀",
    "晋", "蒙", "辽", "吉", "黑",
    "苏", "浙", "京", "闽", "赣",
    "鲁", "豫", "鄂", "湘", "粤",
    "桂", "琼", "川", "贵", "云",
    "西", "陕", "甘", "青", "宁",
    "新"
]


def cv_write(file, frame):
    """
    功能：存储为jpg文件
    :param file: 要存储为jpg的文件
    :param frame: 存储的地址
    :return: 无
    """
    cv.imencode('.jpg', frame)[1].tofile(file)


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
    plate_list = [image.crop(box) for box in box_list]
    return plate_list


def order_points(pts):
    """
    功能: 根据车牌四个顶点构建离散差值数组
    :param pts: 车牌四个顶点坐标
    :return: 离散差值数组
    """
    # 初始化坐标点
    rect = np.zeros((4, 2), dtype='float32')

    # 获取左上角和右下角坐标点
    # 每行像素值进行相加；若axis=0，每列像素值相加
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]

    # 分别计算左上角和右下角的离散差值
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]

    return rect


def four_point_transform(image, pts):
    """
    功能: 根据车牌四个顶点进行车牌区域图像校正
    :param image: 车牌区域图像
    :param pts: 车牌四个顶点坐标
    :return: 校正后的图像
    """
    # 获取坐标点，并将它们分离开来
    rect = order_points(pts)
    (tl, tr, br, bl) = rect

    # 计算新图片的宽度值，选取水平差值的最大值
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))

    # 计算新图片的高度值，选取垂直差值的最大值
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))

    # 构建新图片的4个坐标点,左上角为原点
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")

    # 获取透视变换矩阵并应用它
    m = cv.getPerspectiveTransform(rect, dst)
    # 进行透视变换
    warped = cv.warpPerspective(image, m, (maxWidth, maxHeight))

    # 返回变换后的结果
    return warped


if __name__ == "__main__":
    # 全局变量
    global points
    # 全部车牌号
    count = 0
    # 文件路径
    file_path = 'car_photo'
    for item in os.listdir(os.path.join(file_path)):
        if '.jpg' in item:
            # 1. 提取车牌区域
            # 对于每个包含车牌的jpg文件
            # 图片标号
            k = -1
            count += 1
            img = cv.imdecode(np.fromfile(os.path.join(file_path, item), dtype=np.uint8), -1)

            # 开始提取文件名上的车牌信息
            _, _, bbox, points, label, _, _ = item.split('-')

            # points代表着车牌的四个顶点信息
            points = points.split('_')
            tmp = points
            points = []
            for _ in tmp:
                points.append([int(_.split('&')[0]), int(_.split('&')[1])])
            # points已为包含四个顶点的三维数组

            # 分割单个参数
            label = label.split('_')
            # 中文
            con = CON_LIST[int(label[0])]
            # 英文及数字
            words = [WORDS_LIST[int(_)] for _ in label[1:]]
            # 中英文车牌字符连接
            label = con + ''.join(words)

            # 还原像素位置
            points = np.array(points, dtype=np.float32)
            # 车牌区域图像校正
            warped = four_point_transform(img, points)

            # 车牌区域保存
            # save_path_plate = os.path.join('plate_photo/', label + '.jpg')
            print(label)
            # cv.imencode('.jpg', warped)[1].tofile(save_path_plate)

            # 2. 分割字符
            save_path_chars = 'char_from_plate'
            # 将array转为image
            candidate_plate_image = Image.fromarray(cv.cvtColor(warped, cv.COLOR_BGR2RGB))
            # 裁剪图片
            image_list = cut_image(candidate_plate_image)
            tmp_path = label

            for candidate_char_image in image_list:
                candidate_char_image = cv.cvtColor(np.asarray(candidate_char_image), cv.COLOR_RGB2BGR)
                # 图片预处理：灰度+二值化
                gray_image = cv.cvtColor(candidate_char_image, cv.COLOR_BGR2GRAY)
                is_success, binary_image = cv.threshold(gray_image, 0, 255, cv.THRESH_OTSU)
                # 向内缩进，去除外边框，横不缩，竖缩10
                offset_Y = 10
                # 切片提取内嵌区域
                offset_region = binary_image[offset_Y:-offset_Y, :]
                # 生成工作区域
                working_region = offset_region
                k = k + 1
                img_path = save_path_chars + '/' + tmp_path[k]
                if not os.path.exists(img_path):
                    # 如果文件目录不存在则创建目录
                    os.makedirs(img_path)
                # 读入文件夹
                files = os.listdir(img_path)
                # 统计文件夹中的文件个数
                num_jpg = len(files)
                cv_write(img_path + '/' + str(num_jpg) + '.jpg', working_region)


