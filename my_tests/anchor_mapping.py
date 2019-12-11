import torch
from torch import nn
import numpy as np
# import torch.nn.functional as F

from train.helpers import *
from misc.postprocessing import *


def test(raw_bbox, image, anchors, pred_bbox, highest_confidence_for_prediction, gt_bbox, pos_idx, high_confidence_indeces, indeces_kept_by_nms, size):
    '''
    what we have:
                  - raw_bbox - all model bbox predictions
                  - image: C x H x W tensor
                  - anchors: #anchors x 4 tensor, scaled in [0, 1]: the set of predifined bounding boxes
                  - pred_bbox: #anchors x 4 tensor, in [0, 1] scale: sorted by confidence higher than threshold
                  - highest_confidence_for_prediction: #anchors x 1 tensor -> confidences for each prediction
                  - gt_bbox: tensor with scaled values [0, 1]: the ground truth bboxes of objects in the image
                  - pos_idx: tensor of the indeces of mapped anchors
                  - high_confidence_indeces: - indeces kept by confidence higher than threshold
                  - indeces_kept_by_nms: - indeces after appllying nms
    '''
    matched_anchors = anchors[pos_idx]
    matched_bbox = raw_bbox[pos_idx]

    print("GT BBOXES: ", gt_bbox, gt_bbox.shape)
    plot_bounding_boxes(image, gt_bbox, "GROUND TRUTH", size)

    print("Matched ANCHORS WITH THEIR RESPECTIVE OFFSET PREDICTIONS: ")
    print(matched_anchors, matched_anchors.shape)
    # for i in range(len(matched_anchors)):
    #     cur_anchor_bbox = matched_anchors[i]
    #     cur_pred_bbox = matched_bbox[i]
    #     plot_bounding_boxes(image, cur_anchor_bbox, "ANCHOR", size)
    #     plot_bounding_boxes(image, cur_pred_bbox, "PRED FROM ANCHOR")
    #     print('Confidence for this pair of anchor/pred: ',
    #           highest_confidence_for_prediction[i], size)

    print("Matched Pred BBOXES: ", matched_bbox, matched_bbox.shape)
    print('CONFIDENCES FOR PREDICTED BBOXES: ', highest_confidence_for_prediction)
    plot_bounding_boxes(image, matched_bbox, "PREDICTED (CHEATED) BY THE NETWORK", size)
    plot_bounding_boxes(image, matched_anchors, "MATCHED ANCHORS", size)

    print("THIS IS PRED BBOX KEPT BY CONFIDENCE", pred_bbox,
          pred_bbox.shape)
    plot_bounding_boxes(image, pred_bbox, "ACTUAL MODEL OUTPUTS", size)
    post_nms_predictions = pred_bbox[indeces_kept_by_nms]
    plot_bounding_boxes(
        image, post_nms_predictions, 'Post NMS predictions', size)
    print("THIS IS POST NMS PREDICTIONS", post_nms_predictions,
          post_nms_predictions.shape)
