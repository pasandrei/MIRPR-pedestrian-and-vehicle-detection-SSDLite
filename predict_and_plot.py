from misc.postprocessing import nms, plot_bounding_boxes
from train.helpers import activations_to_bboxes, create_anchors
from train.config import Params
from data import dataloaders
from architectures.models import SSDNet

import cv2
import numpy as np

import torch
import torch.nn as nn


def model_output_pipeline(params_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = Params(params_path)

    if params.model_id == 'ssdnet':
        model = SSDNet.SSD_Head()
    model.to(device)

    checkpoint = torch.load('misc/experiments/{}/model_checkpoint'.format(params.model_id))
    model.load_state_dict(checkpoint['model_state_dict'])
    print('Model loaded successfully')

    _, valid_loader = dataloaders.get_dataloaders(params)
    sig = nn.Sigmoid()

    anchors, grid_sizes = create_anchors()
    anchors, grid_sizes = anchors.to(device), grid_sizes.to(device)

    model.eval()
    with torch.no_grad():
        for (batch_images, batch_targets) in valid_loader:
            batch_images = batch_images.to(device)

            # predictions[0] = B x #anchors x 4
            # predictions[1] = B x #anchors x 3 -> [0.2, 0.1, 0.9], [0.01, 0.01, 0.8]
            # are elements of predictions really on GPU?
            predictions = model(batch_images)

            # move everything to cpu for plotting
            batch_images = batch_images.cpu()
            predictions[0] = [activations_to_bboxes(
                x, anchors, grid_sizes).cpu() for x in predictions[0]]
            predictions[1] = sig(predictions[1]).cpu()

            assert predictions[0][0].is_cuda is False

            for idx in range(len(batch_images)):
                current_image = batch_images[idx]

                current_image_bboxes = batch_targets[0][idx]
                current_image_class_ids = batch_targets[1][idx]

                current_prediction_bboxes = predictions[0][idx]
                current_prediction_class_ids = predictions[1][idx]

                # assert everything here is on CPU
                plot_model_outputs(current_image, current_image_bboxes, current_image_class_ids,
                                   current_prediction_bboxes, current_prediction_class_ids)
                # return
            return


def plot_model_outputs(current_image, current_image_bboxes, current_image_class_ids,
                       current_prediction_bboxes, current_prediction_class_ids):
    """
    ???
    """
    keep_indices = []
    for idx, one_hot_pred in enumerate(current_prediction_class_ids):
        max_confidence, position = one_hot_pred.max(dim=0)
        if position != 2:
            keep_indices.append(idx)

    current_prediction_bboxes = current_prediction_bboxes.numpy()

    current_image = (current_image * 255).numpy().astype(np.uint8)

    # for idx, one_hot_pred in enumerate(current_prediction_class_ids):
    #     if one_hot_pred.max(dim=0)[1] in [0, 1]:
    #         print(idx, one_hot_pred, current_prediction_bboxes[idx]*320)
    #         plot_bounding_boxes(current_image, np.array(
    #             [current_prediction_bboxes[idx]*320]).astype(np.uint16))

    keep_indices = np.array(keep_indices)

    kept_bboxes = current_prediction_bboxes[keep_indices]

    kept_bboxes = (kept_bboxes * 320).astype(int)
    # current_image = (current_image * 255).numpy().astype(np.uint8)

    plot_bounding_boxes(current_image, kept_bboxes)
    print(kept_bboxes)
    post_nms_bboxes = nms(kept_bboxes)
    print(post_nms_bboxes)

    plot_bounding_boxes(current_image, post_nms_bboxes)


model_output_pipeline('misc/experiments/ssdnet/params.json')
