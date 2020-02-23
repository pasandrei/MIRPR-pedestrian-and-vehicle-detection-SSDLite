import torch
import numpy as np
import itertools
import copy

from math import sqrt
from utils.box_computations import jaccard, wh2corners


def map_to_ground_truth(overlaps, gt_bbox, gt_class, params):
    # inspired by fastai course
    """ maps priors to max IOU obj
   returns:
   - gt_bbox_for_matched_anchors: tensor of size matched_priors x 4 - essentially assigning GT bboxes to corresponding highest IOU priors
   - matched_gt_class_ids: tensor of size priors - where each value of the tensor indicates the class id that the priors feature map cell should predict
    """

    # for each object, what is the prior of maximum overlap
    gt_to_prior_overlap, gt_to_prior_idx = overlaps.max(1)

    # for each prior, what is the object of maximum overlap
    prior_to_gt_overlap, prior_to_gt_idx = overlaps.max(0)

    # for priors of max overlap, set a high value to make sure they match
    prior_to_gt_overlap[gt_to_prior_idx] = 1.99

    idx = torch.arange(0, gt_to_prior_idx.size(0), dtype=torch.int64)
    if overlaps.is_cuda:
        idx = idx.to("cuda:0")
    prior_to_gt_idx[gt_to_prior_idx[idx]] = idx

    # for each prior, get the actual id of the class it should predict, unmatched anchors (low IOU) should predict background
    matched_gt_class_ids = gt_class[prior_to_gt_idx]
    pos = prior_to_gt_overlap > params.mapping_threshold
    matched_gt_class_ids[~pos] = 100  # background code

    # for each matched prior, get the bbox it should predict
    raw_matched_bbox = gt_bbox[prior_to_gt_idx]
    pos_idx = torch.nonzero(pos)[:, 0]
    # which of those max values are actually precise enough?
    gt_bbox_for_matched_anchors = raw_matched_bbox[pos_idx]

    # so now we have the GT represented with priors
    return gt_bbox_for_matched_anchors, matched_gt_class_ids, pos_idx


def match(anchors_ltrb, anchors_xywh, gt_bbox, gt_class, params):
    # inspired by fastai course
    """
    Arguments:
        gt_bbox - #obj x 4 tensor - GT bboxes for objects in the cur img
        gt_class - #obj x 1 tensor - class IDs for objects in cur img

    Explanation:
    argmax matching

    Returns:
    #anchors x 4 tensor -> ground truth bbox for each anchor
    #anchor x 1 tensor -> ground truth label for each anchor (anchors with label 100 predict bg)
    """
    # compute IOU for obj x anchor
    overlaps = jaccard(wh2corners(gt_bbox[:, :2], gt_bbox[:, 2:]), anchors_ltrb)

    # map each anchor to the highest IOU obj, gt_idx - ids of mapped objects
    gt_bbox_for_matched_anchors, matched_gt_class_ids, pos_idx = map_to_ground_truth(
        overlaps, gt_bbox, gt_class, params)

    gt_bbox_out = copy.deepcopy(anchors_xywh)
    gt_bbox_out[pos_idx, :] = gt_bbox_for_matched_anchors

    return gt_bbox_out, matched_gt_class_ids


class DefaultBoxes(object):
    # https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD
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
            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j+0.5)/fk[idx], (i+0.5)/fk[idx]
                    self.default_boxes.append((cx, cy, w, h))

        self.dboxes = torch.tensor(self.default_boxes)
        self.dboxes.clamp_(min=0, max=1)
        # For IoU calculation
        self.dboxes_ltrb = self.dboxes.clone()
        self.dboxes_ltrb[:, 0] = self.dboxes[:, 0] - 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 1] = self.dboxes[:, 1] - 0.5 * self.dboxes[:, 3]
        self.dboxes_ltrb[:, 2] = self.dboxes[:, 0] + 0.5 * self.dboxes[:, 2]
        self.dboxes_ltrb[:, 3] = self.dboxes[:, 1] + 0.5 * self.dboxes[:, 3]

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
