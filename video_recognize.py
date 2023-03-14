# 将视频按帧截取并识别
import pathlib
import cv2 as cv
import os
from PIL import Image
import util

# 按比例缩放预设尺寸，参考CCPD数据集图片尺寸
W = 720
H = 1160


def img_resize(image):
    """
    功能：将图片缩放至预设值
    :param image: 要缩放的图片文件
    :return: 缩放后的图片文件
    """
    height, width = image.shape[0], image.shape[1]
    width_new = W
    height_new = H

    if width / height <= width_new / height_new:
        img_new = cv.resize(image, (width_new, int(height * width_new / width)))
    else:
        # w/w_new >= h/h_new, 按照h比例缩放，w_current = w * (h_new/h)
        img_new = cv.resize(image, (int(width * height_new / height), height_new))
    return img_new


def img_crop(image):
    """
    功能：根据预设值裁剪图片
    :param image: 要裁剪的图片
    :return: 裁剪后的图片
    """
    height, width = image.shape[0], image.shape[1]
    x1 = 0
    x2 = 0
    y1 = 0
    y2 = 0
    if height == H:
        x1 = width / 2 - W / 2
        x2 = width / 2 + W / 2
        y1 = 0
        y2 = H
    else:
        y1 = height / 2 - H / 2
        y2 = height / 2 + H / 2
        x1 = 0
        x2 = W
    img_new = image[int(y1):int(y2), int(x1):int(x2)]
    return img_new


def apart_imagefile(video_file, model_res_chn, model_svm_enu, model_mlp_enu, model_res_enu):
    """
    功能：将视频拆分成图片，每20帧识别一次，输出所有识别结果及对应的帧数
    :param video_file: 要拆分的视频文件
    :param model_res_chn: 识别中文的RES模型
    :param model_svm_enu: 识别英文及数字的SVM模型
    :param model_mlp_enu: 识别英文及数字的MLP模型
    :param model_res_enu: 识别英文及数字的RES模型
    :return: 记录帧数及该帧的识别结果的数组
    """
    # 记录第几帧
    frame_times = []
    # 记录该帧下的预测结果
    predict_results = []
    # 帧数计数器
    count = 0
    # 记录前一次检测出结果的帧数
    prev_frame = 0
    # 记录本次有无识别结果
    get_result = False
    print('Start extracting images!')
    while True:
        # 按帧读取，res为读取帧的正确与否，image为每一帧的图像
        res, image = video_file.read()
        # 如果提取完图片，则退出循环
        if not res:
            print('not res , not image')
            break
        # 每隔10帧
        if count % 10 == 0:
            get_result = False
            # 每帧都新建一个记录该帧下分割出的字符集合
            working_regions = []
            # 当前帧下:
            # 修改图片尺寸
            image = img_resize(image)  # 2060*1160
            image = img_crop(image)  # 720*1160
            # 提取车牌区域
            candidate_plates = util.get_candidate_plates_by_sobel(image)
            if len(candidate_plates):
                # 取第一个区域
                car_plate = candidate_plates[0]
                car_plate = Image.fromarray(cv.cvtColor(car_plate, cv.COLOR_BGR2RGB))
                # 切割字符
                image_list = util.cut_image(car_plate)

                # 对于每个分割出的字符进行预处理
                working_regions = util.char_preprocessing(image_list)
                # 对working_regions做预测
                predict_result = util.predict_region(working_regions,
                                                     model_res_chn, model_svm_enu, model_mlp_enu, model_res_enu)
                if not ('0' <= predict_result[1] <= '9'):
                    predict_results.append(predict_result)
                    # 记录帧
                    frame_times.append(count)
                    prev_frame = count
                    get_result = True
            if (not get_result) and (count - prev_frame > 29):
                predict_results.append('未检测到车牌')
                # 记录帧
                frame_times.append(count)
                prev_frame = count
                get_result = True
        count += 1
    # 识别完成，返回识别结果
    print('End of image extraction!')
    return frame_times, predict_results


def main(video_paths):
    """
    功能: 根据视频文件地址进行识别，并返回结果
    :param video_paths: 视频文件相对地址
    :return: 记录帧数的数组，记录对应帧下识别结果的数组
    """
    # 视频文件相对地址
    video_path = video_paths
    # 获取模型
    model_res_chn, model_svm_enu, model_mlp_enu, model_res_enu = util.model_chn_enu()
    # 读取视频文件
    root_dir = os.getcwd() + os.sep + video_path
    root_dir_path = pathlib.Path(root_dir)
    video = os.path.join(root_dir_path)
    use_video = cv.VideoCapture(video)
    # 进行视频识别
    frames, results = apart_imagefile(use_video, model_res_chn, model_svm_enu, model_mlp_enu, model_res_enu)
    # 结果输出
    use_video.release()
    return frames, results
