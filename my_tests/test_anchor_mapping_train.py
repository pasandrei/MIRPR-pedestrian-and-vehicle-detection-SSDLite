import torch
from torch import nn
import numpy as np
# import torch.nn.functional as F

from train.helpers import *
from misc.postprocessing import *


def test(image, anchors, matched_gt_bbox, current_prediction_class_confidences, gt_bbox, pred_bbox, pos_idx):
    '''
    NUMA DOREL STIE CEI ACI
    what we have: - anchors: the set of predifined bounding boxes
                  - gt_bboxes: the ground truth bboxes of objects in the image
                  - using these 2, we want to match those anchors that intersect well with one gt_bbox
                  - DEBUG: plot gt_bboxes and the anchors that have matched to see if this is done correctly
                  - pos_idx is supposed to be the indeces of mapped anchors
                  - matched_gt_bbox is the gt_bbox that each matched anchor has mapped to
    '''

    # first thing first, get the input tensor ready to be plotted
    image = (image * 255).cpu().numpy().astype(np.uint8)
    print('first lines of image: ')
    for idx, line in enumerate(image):
        print(line)
        if idx == 10:
            break

    # same for other variables of interest
    anchors = (anchors.cpu().numpy() * 320).astype(int)
    matched_gt_bbox = (matched_gt_bbox.detach().cpu().numpy()*320).astype(int)
    current_prediction_class_confidences = current_prediction_class_confidences.detach().sigmoid().cpu().numpy()
    gt_bbox = (gt_bbox.cpu().numpy() * 320).astype(int)
    pred_bbox = (pred_bbox.detach().cpu().numpy() * 320).astype(int)
    pos_idx = (pos_idx.cpu().numpy())
    matched_anchors = anchors[pos_idx]

    print("GT BBOXES: ", gt_bbox, gt_bbox.shape)
    plot_bounding_boxes(image, gt_bbox, "GROUND TRUTH")

    print("ALL ANCHORS WITH THEIR RESPECTIVE OFFSET PREDICTIONS: ")
    print(anchors, anchors.shape)
    print(pred_bbox, pred_bbox.shape)
    # for i in range(len(anchors)):
    #     cur_anchor_bbox = anchors[i]
    #     cur_pred_bbox = pred_bbox[i]
    #     plot_bounding_boxes(image, cur_anchor_bbox, "ANCHOR")
    #     plot_bounding_boxes(image, cur_pred_bbox, "PRED FROM ANCHOR")
    #     print('Confidence for this pair of anchor/pred: ', current_prediction_class_confidences[i])

    print("Now let's see correct anchors and their respective predictions")
    print("matched ANCHORS: ", matched_anchors, matched_anchors.shape)
    print("Matched GT BBOXES: ", matched_gt_bbox, matched_gt_bbox.shape)
    plot_bounding_boxes(image, matched_anchors, "CORRECT ANCHORS")

    matched_pred_bbox = pred_bbox[pos_idx]
    print("Matched Pred BBOXES: ", matched_pred_bbox, matched_pred_bbox.shape)
    print('CONFIDENCES FOR PREDICTED BBOXES: ', current_prediction_class_confidences[pos_idx])
    plot_bounding_boxes(image, matched_pred_bbox, "PREDICTED (CHEATED) BY THE NETWORK")


    print("Actual model outputs are: ")
    keep_indices = []
    for index, one_hot_pred in enumerate(current_prediction_class_confidences):
        max_conf = np.amax(one_hot_pred)
        if max_conf > 0.5:
            keep_indices.append(index)

    print("KEPT INDICDES LENGTH: ", len(keep_indices))
    if keep_indices == []:
        pass
    else:
        print("INDECES KEPT BY CONFIDENCE", keep_indices)
        keep_indices = np.array(keep_indices)
        high_confidence_anchors = anchors[keep_indices]
        high_confidence_model_predictions = pred_bbox[keep_indices]
        print("THIS IS PRED BBOX KEPT BY CONFIDENCE", high_confidence_model_predictions, high_confidence_model_predictions.shape)
        plot_bounding_boxes(image, high_confidence_anchors, "HIGH CONFIDENCE ANCHORS")
        plot_bounding_boxes(image, high_confidence_model_predictions, "ACTUAL MODEL OUTPUTS")
