import numpy as np
import torch

from train.config import Params
from general_config import anchor_config
from data import dataloaders
from architectures.models import SSDNet
from visualize import anchor_mapping
from utils.training import load_model, model_setup


def model_output_pipeline(params_path, model_outputs=False, visualize_anchors=False, visualize_anchor_gt_pair=False):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = Params(params_path)

    if params.model_id == 'ssdnet':
        model = model_setup(device, params)
    model.to(device)

    if model_outputs:
        model, _, _ = load_model(model, params)
        model.eval()

    valid_loader = dataloaders.get_dataloaders_test(params)

    with torch.no_grad():
        total_iou, total_maps = 0, np.array([0, 0, 0, 0, 0, 0])
        for batch_idx, (batch_images, batch_targets, images_info) in enumerate(valid_loader):
            if model_outputs:
                batch_images = batch_images.to(device)
                predictions = model(batch_images)
            else:
                n_classes = params.n_classes if params.loss_type == "BCE" else params.n_classes + 1
                predictions = [torch.randn(params.batch_size, anchor_config.total_anchors, 4),
                               torch.randn(params.batch_size, anchor_config.total_anchors, n_classes)]

            for idx in range(len(batch_images)):
                iou, maps = anchor_mapping.test_anchor_mapping(
                    image=batch_images[idx], bbox_predictions=predictions[0][idx], classification_predictions=predictions[1][idx],
                    gt_bbox=batch_targets[0][idx], gt_class=batch_targets[1][idx], image_info=images_info[idx], params=params,
                    model_outputs=model_outputs, visualize_anchors=visualize_anchors, visualize_anchor_gt_pair=visualize_anchor_gt_pair)
                total_iou += iou
                total_maps += maps

            avg = (batch_idx + 1) * params.batch_size
            print("Mean iou so far: ", total_iou / avg)
            print("Mean maps so far: ", total_maps / avg)
            if batch_idx == 10:
                return
