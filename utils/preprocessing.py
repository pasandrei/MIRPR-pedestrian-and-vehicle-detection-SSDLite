import torch
import numpy as np
import itertools

from math import sqrt
from general_config.config import device


class DefaultBoxes(object):
    def __init__(self, fig_size, feat_size, steps, scales, aspect_ratios, scale_xy=0.1, scale_wh=0.2):
        self.feat_size = feat_size
        self.fig_size = fig_size

        self.scale_xy_ = scale_xy
        self.scale_wh_ = scale_wh

        # According to https://github.com/weiliu89/caffe
        # Calculation method slightly different from paper
        self.steps = steps
        self.scales = scales

        fk = fig_size/np.array(steps)
        self.aspect_ratios = aspect_ratios

        self.default_boxes = []
        # size of feature and number of feature
        for idx, sfeat in enumerate(self.feat_size):
            sk1 = scales[idx]/fig_size
            sk2 = scales[idx+1]/fig_size
            sk3 = sqrt(sk1*sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1*sqrt(alpha), sk1/sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))
            for i, j in itertools.product(range(sfeat), repeat=2):
                for w, h in all_sizes:
                    cx, cy = (i+0.5)/fk[idx], (j+0.5)/fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = torch.tensor(self.default_boxes)
        self.dboxes.clamp_(min=0, max=1)

        self.dboxes.to(device)

        # For IoU calculation
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]

        self.dboxes_ltrb.to(device)  # might not be needed depending on what  clone does

    @property
    def scale_xy(self):
        return self.scale_xy_

    @property
    def scale_wh(self):
        return self.scale_wh_

    def __call__(self, order="ltrb"):
        if order == "ltrb":
            return self.dboxes_ltrb
        if order == "xywh":
            return self.dboxes


def dboxes300_coco():
    fig_size = 300
    feat_size = [38, 19, 10, 5, 3, 1]
    steps = [8, 16, 32, 64, 100, 300]
    # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
    scales = [21, 45, 99, 153, 207, 261, 315]
    aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]
    dboxes = DefaultBoxes(fig_size, feat_size, steps, scales, aspect_ratios)
    return dboxes


def prepare_gt(input_img, gt_target):
    '''
    args:
    - input_img: PIL image HxW
    - gt_target: list of gt bbox coords: (x,y,w,h)

    return:
    gt[0] = tensor of bboxes of objects in image scaled [0,1], in (CENTER, w, h) format
    gt[1] = tensor of class ids in image
    '''
    gt_bboxes, gt_classes = [], []
    for obj in gt_target:
        gt_bboxes.append(obj['bbox'])
        gt_classes.append(obj['category_id'])

    gt = [torch.FloatTensor(gt_bboxes), torch.IntTensor(gt_classes)]

    width, height = input_img.size

    for idx, bbox in enumerate(gt[0]):
        new_bbox = [0] * 4
        new_bbox[0] = (bbox[0] + (bbox[2] / 2)) / width
        new_bbox[1] = (bbox[1] + (bbox[3] / 2)) / height
        new_bbox[2] = bbox[2] / width
        new_bbox[3] = bbox[3] / height
        gt[0][idx] = torch.FloatTensor(new_bbox)

    return gt
