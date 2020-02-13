import torch
import numpy as np


def wh2corners_numpy(ctr, wh):
    return np.concatenate([ctr-wh/2, ctr+wh/2], axis=1)


def wh2corners(ctr, wh):
    # taken from fastai
    return torch.cat([ctr-wh/2, ctr+wh/2], dim=1)


def corners_to_wh(prediction_bboxes):
    """
    (x_left, y_left, x_right, y_right) --> (x_left, y_left, width, height)
    """
    prediction_bboxes[:, 2] = prediction_bboxes[:, 2] - prediction_bboxes[:, 0]
    prediction_bboxes[:, 3] = prediction_bboxes[:, 3] - prediction_bboxes[:, 1]

    return prediction_bboxes


def intersect(box_a, box_b):
    # taken from fastai
    """ Returns the intersection of two boxes """
    max_xy = torch.min(box_a[:, None, 2:], box_b[None, :, 2:])
    min_xy = torch.max(box_a[:, None, :2], box_b[None, :, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def box_sz(b):
    # taken from fastai
    """ Returns the box size"""
    return ((b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1]))


def jaccard(box_a, box_b):
    # taken from fastai
    """ Returns the jaccard distance between two boxes"""
    inter = intersect(box_a, box_b)
    union = box_sz(box_a).unsqueeze(1) + box_sz(box_b).unsqueeze(0) - inter
    return inter / union
