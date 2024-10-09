import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from data import voc as cfg
from ..box_utils import match, log_sum_exp

import numpy as np


class MultiBoxLoss(nn.Module):
    def __init__(self, num_classes, overlap_thresh, prior_for_matching,
                 bkg_label, neg_mining, neg_pos, neg_overlap, encode_target,
                 use_gpu=True):
        super(MultiBoxLoss, self).__init__()
        self.use_gpu = use_gpu
        self.num_classes = num_classes
        self.threshold = overlap_thresh
        self.background_label = bkg_label
        self.encode_target = encode_target
        self.use_prior_for_matching = prior_for_matching
        self.do_neg_mining = neg_mining
        self.negpos_ratio = neg_pos
        self.neg_overlap = neg_overlap
        self.variance = cfg['variance']
        self.mask_loss = nn.CrossEntropyLoss()

    def generate_mask(self, targets):
        # targets:长度为batch的列表，每一个元素记录了一张图片中的目标[num_gt, 5] 5--->[xmin, ymin, xmax, ymax, cls_id]这里是相对坐标
        batch_size = len(targets)
        mask_gt = []
        mask_gt1 = np.zeros((batch_size, 1, 50, 50), dtype=np.float32)
        mask_gt2 = np.zeros((batch_size, 1, 25, 25), dtype=np.float32)
        mask_gt3 = np.zeros((batch_size, 1, 13, 13), dtype=np.float32)

        # 遍历每一张图片
        for i in range(batch_size):
            # 遍历当前图片的每一个目标
            for j in range(len(targets[i])):
                bbox_annotation = targets[i][j]  # [5]--->[xmin, ymin, xmax, ymax, cls_id]
                mask1_xmin = mask_gt1.shape[2] * bbox_annotation[0]
                mask1_ymin = mask_gt1.shape[3] * bbox_annotation[1]
                mask1_xmax = mask_gt1.shape[2] * bbox_annotation[2]
                mask1_ymax = mask_gt1.shape[3] * bbox_annotation[3]
                mask1_xmin = max(int(mask1_xmin), 0)
                mask1_ymin = max(int(mask1_ymin), 0)
                mask1_xmax = min(math.ceil(mask1_xmax + 1), mask_gt1.shape[2])
                mask1_ymax = min(math.ceil(mask1_ymax + 1), mask_gt1.shape[3])
                mask_gt1[i, 0, mask1_ymin:mask1_ymax, mask1_xmin:mask1_xmax] = (bbox_annotation[4]+1).cpu()

                mask2_xmin = mask_gt2.shape[2] * bbox_annotation[0]
                mask2_ymin = mask_gt2.shape[3] * bbox_annotation[1]
                mask2_xmax = mask_gt2.shape[2] * bbox_annotation[2]
                mask2_ymax = mask_gt2.shape[3] * bbox_annotation[3]

                mask2_xmin = max(int(mask2_xmin), 0)
                mask2_ymin = max(int(mask2_ymin), 0)
                mask2_xmax = min(math.ceil(mask2_xmax + 1), mask_gt2.shape[2])
                mask2_ymax = min(math.ceil(mask2_ymax + 1), mask_gt2.shape[3])
                mask_gt2[i, 0, mask2_ymin:mask2_ymax, mask2_xmin:mask2_xmax] = (bbox_annotation[4]+1).cpu()

                mask3_xmin = mask_gt3.shape[2] * bbox_annotation[0]
                mask3_ymin = mask_gt3.shape[3] * bbox_annotation[1]
                mask3_xmax = mask_gt3.shape[2] * bbox_annotation[2]
                mask3_ymax = mask_gt3.shape[3] * bbox_annotation[3]

                mask3_xmin = max(int(mask3_xmin), 0)
                mask3_ymin = max(int(mask3_ymin), 0)
                mask3_xmax = min(math.ceil(mask3_xmax + 1), mask_gt3.shape[2])
                mask3_ymax = min(math.ceil(mask3_ymax + 1), mask_gt3.shape[3])
                mask_gt3[i, 0, mask3_ymin:mask3_ymax, mask3_xmin:mask3_xmax] = (bbox_annotation[4]+1).cpu()

        mask_gt1 = torch.from_numpy(mask_gt1)
        mask_gt2 = torch.from_numpy(mask_gt2)
        mask_gt3 = torch.from_numpy(mask_gt3)

        mask_gt1.type(torch.int64)

        mask_gt.append(mask_gt1)
        mask_gt.append(mask_gt2)
        mask_gt.append(mask_gt3)
        return mask_gt

    def forward(self, predictions, targets):
        loc_data, conf_data, priors, mask = predictions
        batch_size = len(targets)
        num_classes = self.num_classes
        mask_gt = self.generate_mask(targets)

        total_pixel = []
        total_mask = []
        for (m, m_g) in zip(mask, mask_gt):
            m = m.view(batch_size, num_classes, -1)  # [8, 11, 50x50]
            m = m.transpose(1, 2).contiguous()  # [8, 50x50, 11]
            m = m.view(-1, num_classes)  # [8x50x50, 11]
            total_pixel.append(m)
            m_g = m_g.view(batch_size, -1).contiguous()  # [8, 50x50]
            m_g = m_g.view(-1)  # [8 x 50x50]
            total_mask.append(m_g)

        total_pixel, total_mask = torch.cat(total_pixel, 0).contiguous(), torch.cat(total_mask, 0).contiguous()
        total_mask = torch.tensor(total_mask, dtype=torch.int64)
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        loss_m = self.mask_loss(total_pixel, total_mask.to(device))

        num = loc_data.size(0)
        priors = priors[:loc_data.size(1), :]
        num_priors = (priors.size(0))

        loc_t = torch.Tensor(num, num_priors, 4)
        conf_t = torch.LongTensor(num, num_priors)
        for idx in range(num):
            truths = targets[idx][:, :-1].data
            labels = targets[idx][:, -1].data
            defaults = priors.data
            match(self.threshold, truths, defaults, self.variance, labels,
                  loc_t, conf_t, idx)
        if self.use_gpu:
            loc_t = loc_t.cuda()
            conf_t = conf_t.cuda()

        loc_t = Variable(loc_t, requires_grad=False)
        conf_t = Variable(conf_t, requires_grad=False)

        pos = conf_t > 0
        num_pos = pos.sum(dim=1, keepdim=True)

        pos_idx = pos.unsqueeze(pos.dim()).expand_as(loc_data)
        loc_p = loc_data[pos_idx].view(-1, 4)
        loc_t = loc_t[pos_idx].view(-1, 4)
        loss_l = F.smooth_l1_loss(loc_p, loc_t, size_average=False)

        batch_conf = conf_data.view(-1, self.num_classes)
        loss_c = log_sum_exp(batch_conf) - batch_conf.gather(1, conf_t.view(-1, 1))

        loss_c = loss_c.view(num, -1)  # The line added   # 新添加
        loss_c[pos] = 0  # filter out pos boxes for now
        loss_c = loss_c.view(num, -1)
        _, loss_idx = loss_c.sort(1, descending=True)
        _, idx_rank = loss_idx.sort(1)
        num_pos = pos.long().sum(1, keepdim=True)
        num_neg = torch.clamp(self.negpos_ratio*num_pos, max=pos.size(1)-1)
        neg = idx_rank < num_neg.expand_as(idx_rank)

        pos_idx = pos.unsqueeze(2).expand_as(conf_data)
        neg_idx = neg.unsqueeze(2).expand_as(conf_data)
        conf_p = conf_data[(pos_idx+neg_idx).gt(0)].view(-1, self.num_classes)
        targets_weighted = conf_t[(pos+neg).gt(0)]
        loss_c = F.cross_entropy(conf_p, targets_weighted, size_average=False)

        N = num_pos.data.sum().float()
        loss_l /= N
        loss_c /= N
        loss_l *= 2
        loss_c *= 2
        return loss_l, loss_c, loss_m
