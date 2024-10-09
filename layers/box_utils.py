# -*- coding: utf-8 -*-
import torch
def point_form(boxes):
    return torch.cat((boxes[:, :2] - boxes[:, 2:]/2,     # xmin, ymin
                     boxes[:, :2] + boxes[:, 2:]/2), 1)  # xmax, ymax


def center_size(boxes):
    return torch.cat((boxes[:, 2:] + boxes[:, :2])/2,  # cx, cy
                     boxes[:, 2:] - boxes[:, :2], 1)  # w, h


def box_area(boxes):
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


# compute IOU
# compute intersect
def intersect(box_a, box_b):
    A = box_a.size(0)
    B = box_b.size(0)
    max_xy = torch.min(box_a[:, 2:].unsqueeze(1).expand(A, B, 2),
                       box_b[:, 2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(box_a[:, :2].unsqueeze(1).expand(A, B, 2),
                       box_b[:, :2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]

def jaccard(box_a, box_b):
    inter = intersect(box_a, box_b)
    area_a = ((box_a[:, 2]-box_a[:, 0]) *
              (box_a[:, 3]-box_a[:, 1])).unsqueeze(1).expand_as(inter)  # [A,B]
    area_b = ((box_b[:, 2]-box_b[:, 0]) *
              (box_b[:, 3]-box_b[:, 1])).unsqueeze(0).expand_as(inter)  # [A,B]
    union = area_a + area_b - inter
    return inter / union  # [A,B]


def calc_nwd_tensor(bboxes1, bboxes2, eps=1e-6, constant=150):
    temp_bboxes1 = bboxes1 * 400
    temp_bboxes2 = bboxes2 * 400
    area1 = (temp_bboxes1[..., 2] - temp_bboxes1[..., 0]) * (temp_bboxes1[..., 3] - temp_bboxes1[..., 1])
    area2 = (temp_bboxes2[..., 2] - temp_bboxes2[..., 0]) * (temp_bboxes2[..., 3] - temp_bboxes2[..., 1])

    area1 = area1.unsqueeze(-1)  # [num_gt, 1]
    area2 = area2.unsqueeze(0)  # [1, num_anchors]

    temp1 = torch.sqrt(area1)
    temp2 = torch.sqrt(area2)
    constant = (temp1 + temp2 + eps) / 2  # [num_gt, num_anchors]
    eps = torch.tensor([eps])

    center1 = (temp_bboxes1[..., :, None, :2] + temp_bboxes1[..., :, None, 2:]) / 2
    center2 = (temp_bboxes2[..., None, :, :2] + temp_bboxes2[..., None, :, 2:]) / 2
    whs = center1[..., :2] - center2[..., :2]

    center_distance = whs[..., 0] * whs[..., 0] + whs[..., 1] * whs[..., 1] + eps

    w1 = temp_bboxes1[..., :, None, 2] - temp_bboxes1[..., :, None, 0] + eps
    h1 = temp_bboxes1[..., :, None, 3] - temp_bboxes1[..., :, None, 1] + eps
    w2 = temp_bboxes2[..., None, :, 2] - temp_bboxes2[..., None, :, 0] + eps
    h2 = temp_bboxes2[..., None, :, 3] - temp_bboxes2[..., None, :, 1] + eps

    wh_distance = ((w1 - w2) ** 2 + (h1 - h2) ** 2) / 4

    wassersteins = torch.sqrt(center_distance + wh_distance)
    normalized_wassersteins = torch.exp(-wassersteins/constant)

    return normalized_wassersteins


def match(threshold, truths, priors, variances, labels, loc_t, conf_t, idx):

    temp_bboxes = truths * 400
    area = box_area(temp_bboxes)
    small_gt = area <= (32 * 32)
    not_small_gt = area > (32 * 32)
    final_ious = torch.zeros((truths.size(0), priors.size(0)))
    nwds = calc_nwd_tensor(truths, point_form(priors))

    overlaps = jaccard(truths, point_form(priors))  # point_form()将xywh转换为xmin,ymin, xmax, ymax

    final_ious[small_gt] = nwds[small_gt]
    final_ious[not_small_gt] = overlaps[not_small_gt]

    best_prior_overlap, best_prior_idx = final_ious.max(1, keepdim=True)  # [num_gt, 1]
    best_truth_overlap, best_truth_idx = final_ious.max(0, keepdim=True)  # [1, 20194]
    best_truth_idx.squeeze_(0)  # [20194]
    best_truth_overlap.squeeze_(0)  # [20194]
    best_prior_idx.squeeze_(1)  # [num_gt]
    best_prior_overlap.squeeze_(1)  # [num_gt]
    best_truth_overlap.index_fill_(0, best_prior_idx, 2)  # ensure best prior
    # TODO refactor: index  best_prior_idx with long tensor
    for j in range(best_prior_idx.size(0)):
        best_truth_idx[best_prior_idx[j]] = j
    matches = truths[best_truth_idx]          # Shape: [num_priors,4]  每一个anchor匹配到的gt的坐标
    conf = labels[best_truth_idx] + 1         # Shape: [num_priors]  # Shape: [num_priors]，+1  0:back
    conf[best_truth_overlap < threshold] = 0  # anchor与gt的最大iou小于阈值认为这个anchor是负样本  label as background
    loc = encode(matches, priors, variances)  # encode(matches,priors,variances)
    loc_t[idx] = loc    # [num_priors,4] encoded offsets to learn
    conf_t[idx] = conf  # [num_priors] top class label for each prior

def encode(matched, priors, variances):
    g_cxcy = (matched[:, :2] + matched[:, 2:])/2 - priors[:, :2]
    g_cxcy /= (variances[0] * priors[:, 2:])
    g_wh = (matched[:, 2:] - matched[:, :2]) / priors[:, 2:]
    g_wh = torch.log(g_wh) / variances[1]
    return torch.cat([g_cxcy, g_wh], 1)  # [num_priors,4]

def decode(loc, priors, variances):
    boxes = torch.cat((
        priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:],
        priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])), 1)
    boxes[:, :2] -= boxes[:, 2:] / 2
    boxes[:, 2:] += boxes[:, :2]
    return boxes


def log_sum_exp(x):
    x_max = x.data.max()
    return torch.log(torch.sum(torch.exp(x-x_max), 1, keepdim=True)) + x_max

def nms(boxes, scores, overlap=0.5, top_k=200):
    keep = scores.new(scores.size(0)).zero_().long()
    if boxes.numel() == 0:
        return keep
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]
    area = torch.mul(x2 - x1, y2 - y1)
    v, idx = scores.sort(0)  # sort in ascending order
    idx = idx[-top_k:]  # indices of the top-k largest vals
    xx1 = boxes.new()
    yy1 = boxes.new()
    xx2 = boxes.new()
    yy2 = boxes.new()
    w = boxes.new()
    h = boxes.new()
    count = 0
    while idx.numel() > 0:
        i = idx[-1]  # index of current largest val
        # keep.append(i)
        keep[count] = i
        count += 1
        if idx.size(0) == 1:
            break
        idx = idx[:-1]  # remove kept element from view
        # load bboxes of next highest vals

        idx = torch.autograd.Variable(idx, requires_grad=False)
        idx = idx.data
        x1 = torch.autograd.Variable(x1, requires_grad=False)
        x1 = x1.data
        y1 = torch.autograd.Variable(y1, requires_grad=False)
        y1 = y1.data
        x2 = torch.autograd.Variable(x2, requires_grad=False)
        x2 = x2.data
        y2 = torch.autograd.Variable(y2, requires_grad=False)
        y2 = y2.data
        torch.index_select(x1, 0, idx, out=xx1)
        torch.index_select(y1, 0, idx, out=yy1)
        torch.index_select(x2, 0, idx, out=xx2)
        torch.index_select(y2, 0, idx, out=yy2)

        if area[i] * 400 * 400 < 32 * 32:
            nwds = calc_nwd_tensor(boxes[i].unsqueeze(0), boxes[idx]).squeeze(0)
            idx = idx[nwds.le(overlap)]
        else:
            # store element-wise max with next highest score
            xx1 = torch.clamp(xx1, min=x1[i])
            yy1 = torch.clamp(yy1, min=y1[i])
            xx2 = torch.clamp(xx2, max=x2[i])
            yy2 = torch.clamp(yy2, max=y2[i])
            w.resize_as_(xx2)
            h.resize_as_(yy2)
            w = xx2 - xx1
            h = yy2 - yy1
            # check sizes of xx1 and xx2.. after each iteration
            w = torch.clamp(w, min=0.0)
            h = torch.clamp(h, min=0.0)
            inter = w * h
            # IoU = i / (area(a) + area(b) - i)
            # 否者出错RuntimeError: index_select(): functions with out=... arguments don't support automatic differentiation, but one of the arguments requires grad.
            area = torch.autograd.Variable(area, requires_grad=False)
            area = area.data
            idx = torch.autograd.Variable(idx, requires_grad=False)
            idx = idx.data
            rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
            union = (rem_areas - inter) + area[i]
            IoU = inter / union  # store result in iou
            # keep only elements with an IoU <= overlap
            '''4.保留iou值小于nms阈值的预测边界框的索引'''
            idx = idx[IoU.le(overlap)]  # 保留交并比小于阈值的预测边界框的id

        """
        # store element-wise max with next highest score
        xx1 = torch.clamp(xx1, min=x1[i])
        yy1 = torch.clamp(yy1, min=y1[i])
        xx2 = torch.clamp(xx2, max=x2[i])
        yy2 = torch.clamp(yy2, max=y2[i])
        w.resize_as_(xx2)
        h.resize_as_(yy2)
        w = xx2 - xx1
        h = yy2 - yy1
        # check sizes of xx1 and xx2.. after each iteration
        w = torch.clamp(w, min=0.0)
        h = torch.clamp(h, min=0.0)
        inter = w * h
        # IoU = i / (area(a) + area(b) - i)
        # 否者出错RuntimeError: index_select(): functions with out=... arguments don't support automatic differentiation, but one of the arguments requires grad.
        area = torch.autograd.Variable(area, requires_grad=False)
        area = area.data
        idx = torch.autograd.Variable(idx, requires_grad=False)
        idx = idx.data
        rem_areas = torch.index_select(area, 0, idx)  # load remaining areas)
        union = (rem_areas - inter) + area[i]
        IoU = inter / union  # store result in iou
        # keep only elements with an IoU <= overlap
        '''4.保留iou值小于nms阈值的预测边界框的索引'''
        idx = idx[IoU.le(overlap)]  # 保留交并比小于阈值的预测边界框的id
        """

    return keep, count