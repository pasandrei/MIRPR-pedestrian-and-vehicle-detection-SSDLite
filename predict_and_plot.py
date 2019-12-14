from misc.postprocessing import nms, plot_bounding_boxes
from train.helpers import *
from train.config import Params
from data import dataloaders
from architectures.models import SSDNet
from my_tests import anchor_mapping
from misc.model_output_handler import *

import cv2
import numpy as np

import torch
import torch.nn as nn


def model_output_pipeline(params_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    params = Params(params_path)

    if params.model_id == 'ssdnet':
        model = SSDNet.SSD_Head(params.n_classes)
    model.to(device)

    checkpoint = torch.load('misc/experiments/{}/model_checkpoint'.format(params.model_id))
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model loaded successfully')
    #
    _, valid_loader = dataloaders.get_dataloaders(params)
    output_handler = Model_output_handler()
    print('EVALUATION')
    model.eval()
    with torch.no_grad():
        total_iou, total_maps = 0, np.array([0, 0, 0, 0, 0, 0])
        for batch_idx, (batch_images, batch_targets, images_info) in enumerate(valid_loader):
            batch_images = batch_images.to(device)
            # predictions[0] = B x  # anchors x 4
            # predictions[1] = B x  # anchors x 2 -> [0.2, 0.9], [0.01, 0.01]
            predictions = model(batch_images)
            # predictions = [torch.FloatTensor(
            #     params.batch_size, 400*12 + 100*20 + 25*30 + 9*40 + 4*40 + 1*50, 4),
            #     torch.FloatTensor(params.batch_size, 400*12 + 100*20 + 25*30 + 9*40 + 4*40 + 1*50, 2)]

            for idx in range(len(batch_images)):
                complete_outputs = output_handler.process_outputs(
                    predictions[0][idx], predictions[1][idx], images_info[idx])

                print("THIS IS HOW COMPLETE OUTPUTS look like", complete_outputs)

                iou, maps = anchor_mapping.test_anchor_mapping(
                    image=batch_images[idx], bbox_predictions=predictions[0][idx], classification_predictions=predictions[1][idx],
                    gt_bbox=batch_targets[0][idx], gt_class=batch_targets[1][idx], image_info=images_info[idx])
                total_iou += iou
                total_maps += maps

            avg = (batch_idx + 1) * params.batch_size
            print("Mean iou so far: ", total_iou / avg)
            print("Mean maps so far: ", total_maps / avg)
            if batch_idx == 10:
                return


# model_output_pipeline('misc/experiments/ssdnet/params.json')
