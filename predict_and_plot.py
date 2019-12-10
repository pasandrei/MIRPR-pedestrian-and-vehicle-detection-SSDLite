from misc.postprocessing import nms, plot_bounding_boxes
from train.helpers import *
from train.config import Params
from data import dataloaders
from architectures.models import SSDNet
from my_tests import test_anchor_mapping

import cv2
import numpy as np

import torch
import torch.nn as nn


def model_output_pipeline(params_path):
    #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    device = torch.device("cpu")
    params = Params(params_path)

    if params.model_id == 'ssdnet':
        model = SSDNet.SSD_Head(params.n_classes)
    model.to(device)

    checkpoint = torch.load('misc/experiments/{}/model_checkpoint'.format(params.model_id))
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model loaded successfully')

    _, valid_loader = dataloaders.get_dataloaders(params)

    anchors, grid_sizes = create_anchors()
    anchors, grid_sizes = anchors.to(device), grid_sizes.to(device)

    print('EVALUATION')
    model.eval()
    with torch.no_grad():
        for (batch_images, batch_targets, images_info) in valid_loader:
            batch_images = batch_images.to(device)
            # predictions[0] = B x #anchors x 4
            # predictions[1] = B x #anchors x 2 -> [0.2, 0.9], [0.01, 0.01]
            predictions = model(batch_images)

            for idx in range(len(batch_images)):
                gt_bbox, gt_class = batch_targets[0][idx].to(
                    device), batch_targets[1][idx].to(device)

                # make network outputs same as gt bbox format
                pred_bbox = activations_to_bboxes(predictions[0][idx], anchors, grid_sizes)

                corner_anchors = hw2corners(anchors[:, :2], anchors[:, 2:])
                overlaps = jaccard(gt_bbox, corner_anchors)

                # map each anchor to the highest IOU obj, gt_idx - ids of mapped objects
                gt_bbox_for_matched_anchors, matched_gt_class_ids, matched_pred_class, pos_idx = map_to_ground_truth(
                    overlaps, gt_bbox, gt_class, pred_bbox)

                # thorough prints for debugging, or just model outputs
                test_anchor_mapping.test(batch_images[idx], corner_anchors, gt_bbox_for_matched_anchors,
                                         pred_bbox, predictions[1][idx], gt_bbox, pos_idx, just_outputs=0)
            return

# model_output_pipeline('misc/experiments/ssdnet/params.json')
