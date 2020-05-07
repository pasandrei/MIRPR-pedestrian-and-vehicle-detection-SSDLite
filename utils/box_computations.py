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


def get_intersection(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    return np.array([x1, y1, x2, y2])


def get_bbox_area(bbox):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    if width < 0 or height < 0:
        return 0

    return width*height


def get_IoU(bbox1, bbox2):
    bbox1_x1, bbox1_y1, bbox1_x2, bbox1_y2 = bbox1
    bbox2_x1, bbox2_y1, bbox2_x2, bbox2_y2 = bbox2

    intersection = get_intersection(bbox1, bbox2)
    intersection_area = get_bbox_area(intersection)

    union_area = get_bbox_area(bbox1) + get_bbox_area(bbox2) - intersection_area

    # want to eliminate invalid boxes
    if union_area == 0:
        return 1

    return intersection_area/union_area
