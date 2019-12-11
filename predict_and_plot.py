from misc.postprocessing import nms, plot_bounding_boxes
from train.helpers import *
from train.config import Params
from data import dataloaders
from architectures.models import SSDNet
from misc.model_output_handler import *

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
    output_handler = Model_output_handler(device)

    print('EVALUATION')
    model.eval()
    with torch.no_grad():
        for (batch_images, batch_targets, images_info) in valid_loader:
            batch_images = batch_images.to(device)
            # predictions[0] = B x #anchors x 4
            # predictions[1] = B x #anchors x 2 -> [0.2, 0.9], [0.01, 0.01]
            predictions = model(batch_images)

            for idx in range(len(batch_images)):
                output_handler.test_anchor_mapping(
                    batch_images[idx], predictions[0][idx], predictions[1][idx], batch_targets[0][idx], batch_targets[1][idx], images_info[idx])

            return

# model_output_pipeline('misc/experiments/ssdnet/params.json')
