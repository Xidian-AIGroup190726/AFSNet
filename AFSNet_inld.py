import colorsys
import os
import time
import warnings
from torch.autograd import Variable

import numpy as np
import torch
from data import voc
from SMENet import build_SMENet
from data import BaseTransform

import torch.backends.cudnn as cudnn
from PIL import Image, ImageDraw, ImageFont
from layers import *
import cv2
from utils.utils_map import get_classes

warnings.filterwarnings("ignore")


# --------------------------------------------#
#   使用自己训练好的模型预测需要修改3个参数
#   model_path、backbone和classes_path都需要修改！
#   如果出现shape不匹配
#   一定要注意训练时的config里面的num_classes、
#   model_path和classes_path参数的修改
# --------------------------------------------#
class SMENet_inld(object):
    _defaults = {
        # --------------------------------------------------------------------------#
        #   使用自己训练好的模型进行预测一定要修改model_path和classes_path！
        #   model_path指向logs文件夹下的权值文件，classes_path指向model_data下的txt
        #
        #   训练好后logs文件夹下存在多个权值文件，选择验证集损失较低的即可。
        #   验证集损失较低不代表mAP较高，仅代表该权值在验证集上泛化性能较好。
        #   如果出现shape不匹配，同时要注意训练时的model_path和classes_path参数的修改
        # --------------------------------------------------------------------------#
        "model_path": 'weights/SME4004.pth',
        "classes_path": 'model_data/nwpu_classes.txt',
        # ---------------------------------------------------------------------#
        #   用于预测的图像大小，和train时使用同一个即可
        # ---------------------------------------------------------------------#
        "input_shape": [400, 400],
        # ---------------------------------------------------------------------#
        #   只有得分大于置信度的预测框会被保留下来
        # ---------------------------------------------------------------------#
        "confidence": 0.5,
        # ---------------------------------------------------------------------#
        #   非极大抑制所用到的nms_iou大小
        # ---------------------------------------------------------------------#
        "nms_iou": 0.45,
        # ---------------------------------------------------------------------#
        #   用于指定先验框的大小
        # ---------------------------------------------------------------------#
        'anchors_size': [25, 65, 116, 167, 218, 269, 320],
        # -------------------------------#
        #   是否使用Cuda
        #   没有GPU可以设置成False
        # -------------------------------#
        "cuda": True,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    # ---------------------------------------------------#
    #   初始化ssd
    # ---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
        # ---------------------------------------------------#
        #   计算总的类的数量
        # ---------------------------------------------------#
        self.class_names, self.num_classes = get_classes(self.classes_path)
        self.anchors = PriorBox(voc).forward()
        if self.cuda:
            self.anchors = self.anchors.cuda()
        self.num_classes = self.num_classes + 1

        self.generate()

    def generate(self):
        #   载入模型与权值
        self.net = build_SMENet('test', 400, self.num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/" + image_id + ".txt"), "w")
        # ---------------------------------------------------------#
        #   在这里将图像转换成RGB图像，防止灰度图在预测时报错。
        #   代码仅仅支持RGB图像的预测，所有其它类型的图像都会转化成RGB
        # ---------------------------------------------------------#
        image_path = os.path.join("./VOCNWPU/VOC2012/JPEGImages/" + image_id + ".jpg")
        image = cv2.imread(image_path, cv2.IMREAD_COLOR)
        x = torch.from_numpy(BaseTransform(400, (86, 91, 82))(image)[0]).permute(2, 0, 1)
        x = Variable(x.unsqueeze(0))

        with torch.no_grad():
            y = self.net(x)  # [1, 11, 200, 5]forward pass
            detections = y.data  # [1, 11, 200, 5]
            # scale each detection back up to the image
            scale = torch.Tensor([image.shape[1], image.shape[0],
                                  image.shape[1], image.shape[0]])
            bboxes_out = []  # ndarray:[n, 4]
            labels_out = []  # ndarray:[n,]
            scores_out = []  # ndarray:[n,]

            for i in range(detections.size(1)):  # 遍历每一个类别
                j = 0
                while j <= 199 and detections[0, i, j, 0] > self.confidence:
                    scores_out.append(detections[0, i, j, 0])
                    labels_out.append(i)
                    pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                    coords = (pt[0], pt[1], pt[2], pt[3])
                    bboxes_out.append(coords)
                    j += 1

            bboxes_out = np.array(bboxes_out)
            labels_out = np.array(labels_out)
            scores_out = np.array(scores_out)
            if len(bboxes_out) == 0:
                print("没有检测到任何目标!")

        for i, c in list(enumerate(labels_out)):
            predicted_class = self.class_names[int(c)-1]
            box = bboxes_out[i]
            score = str(scores_out[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (
            predicted_class, score[:6], str(int(top)), str(int(left)), str(int(bottom)), str(int(right))))

        f.close()
        return
