from __future__ import division, print_function, absolute_import

from timeit import time
import warnings
import cv2
import os
import numpy as np
from PIL import Image
from yolo import YOLO

from deep_sort import preprocessing
from deep_sort import nn_matching
from deep_sort.detection import Detection
# from deep_sort.tracker import Tracker
from tools import generate_detections as gdet

warnings.filterwarnings('ignore')


def write_txt(name, msg):
    # path_name = 'D:/Desktop/GitHub/ObstacleDetection/'
    full_path_name = name + '.txt'
    if not os.path.exists(full_path_name):
        file = open(full_path_name, 'w')
    file = open(full_path_name, 'a')
    file.write(msg)


def main(yolo):
    # Definition of the parameters
    max_cosine_distance = 0.3
    nn_budget = None
    nms_max_overlap = 1.0

    # Deep SORT
    model_filename = 'model_data/mars-small128.pb'
    encoder = gdet.create_box_encoder(model_filename, batch_size=1)

    metric = nn_matching.NearestNeighborDistanceMetric("cosine", max_cosine_distance, nn_budget)
    # tracker = Tracker(metric)

    fps = 0.0

    # %% 循环开始
    dataname = "man1_short_2"
    path = "D:/Desktop/GitHub/Multi-task/examples/ExpKITTI_joint/"

    for root, dirs, files in os.walk(path + dataname):
        for file in files:
            img_name = file.split('.png')  # no use
            img_path = os.path.join(root, file)
            frame = np.array(Image.open(img_path))

            t1 = time.time()  # 返回当前时间的时间戳

            image = Image.fromarray(frame[..., ::-1])  # 实现array到image的转换  # bgr to rgb
            boxs = yolo.detect_image(image)[0]
            confidence = yolo.detect_image(image)[1]

            features = encoder(frame, boxs)  # 以encoding指定的编码格式编码字符串

            detections = [Detection(bbox, confidence, feature) for bbox, confidence, feature in
                          zip(boxs, confidence, features)]

            # Run non-maxima suppression.
            boxes = np.array([d.tlwh for d in detections])
            scores = np.array([d.confidence for d in detections])
            indices = preprocessing.non_max_suppression(boxes, nms_max_overlap, scores)
            detections = [detections[i] for i in indices]

            # 检测框信息
            # 这里是描述框子的输出信息
            for det in detections:
                bbox = det.to_tlbr()
                cv2.rectangle(frame, (int(bbox[0]), int(bbox[1])), (int(bbox[2]), int(bbox[3])), (255, 255, 0), 2)
                center_x = int((int(bbox[0]) + int(bbox[2])) / 2)
                center_y = int((int(bbox[1]) + int(bbox[3])) / 2)
                point = ' [' + img_name[0] + ',' + str(center_x) + ',' + str(center_y) + ']'
                txt = point
                cv2.putText(frame, txt, (int(bbox[0]), int(bbox[1])), 0, 0.5, (255, 255, 255), 1)
                points_lefttop_txt = img_name[0] + ',' + str(int(bbox[0])) + ',' + str(int(bbox[2]))
                points_rightbuttom_txt = img_name[0] + ',' + str(int(bbox[1])) + ',' + str(int(bbox[3]))
                write_txt('points_lefttop', points_lefttop_txt + '\n')
                write_txt('points_rightbuttom', points_rightbuttom_txt + '\n')

                width = int(bbox[2]) - int(bbox[0])
                height = int(bbox[1]) - int(bbox[3])
                points_w_h_txt = str(width) + ',' + str(height)
                write_txt('points_w_l', points_w_h_txt + '\n')
                print(point)

            cv2.imshow('', frame)

            kuang_dir = path + dataname + "_kuang/"
            if not os.path.exists(kuang_dir):
                os.makedirs(kuang_dir)
            savename = kuang_dir + file
            cv2.imwrite(savename, frame)

            fps = (fps + (1. / (time.time() - t1))) / 2
            print("FPS = %f" % fps)

            # Press Q to stop!
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


if __name__ == '__main__':
    main(YOLO())
