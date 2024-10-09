from __future__ import print_function
import matplotlib.pyplot as plt
import json
import os
import cv2
import argparse
import numpy as np
import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable
from data import VOC_ROOT
from PIL import Image
from data import VOCAnnotationTransform, VOCDetection, BaseTransform, VOC_CLASSES
from SMENet import build_SMENet
from draw_box_utils import draw_objs


parser = argparse.ArgumentParser(description='SMENet Detection')
parser.add_argument('--trained_model', default='weights/SME4004.pth',
                    type=str, help='Trained state_dict file path to open')
parser.add_argument('--save_folder', default='predict/', type=str,
                    help='Dir to save results')
parser.add_argument('--visual_threshold', default=0.6, type=float,
                    help='Final confidence threshold')
parser.add_argument('--cuda', default=False, type=bool,
                    help='Use cuda to train model')
parser.add_argument('--voc_root', default=VOC_ROOT, help='Location of VOC root directory')
parser.add_argument('-f', default=None, type=str, help="Dummy arg so we can load in Jupyter Notebooks")
args = parser.parse_args()

if args.cuda and torch.cuda.is_available():
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def test_net(net, cuda, transform):
    # dump predictions and assoc. ground truth to text file for now
    img_path = "./VOCNWPU/VOC2012/JPEGImages/350.jpg"
    # print(filename)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)
    x = torch.from_numpy(transform(img)[0]).permute(2, 0, 1)
    x = Variable(x.unsqueeze(0))

    if cuda:
        x = x.cuda()

    with torch.no_grad():
        y = net(x)  # [1, 11, 200, 5]forward pass
        detections = y.data  # [1, 11, 200, 5]
        # scale each detection back up to the image
        scale = torch.Tensor([img.shape[1], img.shape[0],
                             img.shape[1], img.shape[0]])
        pred_num = 0
        bboxes_out = []  # ndarray:[n, 4]
        labels_out = []  # ndarray:[n,]
        scores_out = []  # ndarray:[n,]

        for i in range(detections.size(1)):  # 遍历每一个类别
            j = 0
            while detections[0, i, j, 0] >= 0.6:
                scores_out.append(detections[0, i, j, 0])
                labels_out.append(i)
                pt = (detections[0, i, j, 1:] * scale).cpu().numpy()
                coords = (pt[0], pt[1], pt[2], pt[3])
                bboxes_out.append(coords)
                pred_num += 1
                j += 1
        bboxes_out = np.array(bboxes_out)
        labels_out = np.array(labels_out)
        scores_out = np.array(scores_out)
        if len(bboxes_out) == 0:
            print("没有检测到任何目标!")

        original_img = Image.open(img_path)
        json_path = "./pascal_voc_classes.json"
        json_file = open(json_path, 'r')
        class_dict = json.load(json_file)
        category_index = {str(v): str(k) for k, v in class_dict.items()}

        plot_img = draw_objs(original_img,
                             bboxes_out,
                             labels_out,
                             scores_out,
                             category_index=category_index,
                             box_thresh=0.5,
                             line_thickness=3,
                             font='arial.ttf',
                             font_size=20)

        plt.imshow(plot_img)
        plt.show()


def test_voc():
    # load net
    num_classes = len(VOC_CLASSES) + 1  # +1 background
    net = build_SMENet('test', 400, num_classes)
    net.load_state_dict(torch.load(args.trained_model))   # load pretrained model
    net.eval()
    print('Finished loading model!')
    # load data
    if args.cuda:
        net = net.cuda()
        cudnn.benchmark = True
    # evaluation
    test_net(net, args.cuda, BaseTransform(net.size, (86, 91, 82)),)


if __name__ == '__main__':
    test_voc()
