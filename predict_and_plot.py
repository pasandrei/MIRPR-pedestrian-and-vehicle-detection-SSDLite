import numpy as np
import torch

from train.params import Params
from general_config import anchor_config, constants, classes_config, general_config
from data import dataloaders
from visualize import anchor_mapping
from utils.training import load_weigths_only, model_setup
from general_config.general_config import device


def model_output_pipeline(model_outputs=True, visualize_anchors=False,
                          visualize_anchor_gt_pair=False, verbose=False, very_verbose=False):
    """
    model_outputs - flag to enable plotting model outputs
    visualize_anchors - flag to visualize anchors
    visualize_anchor_gt_pair - flag to visualize ground truth bboxes and respective anchors
    """
    params = Params(constants.params_path.format(general_config.model_id))

    if model_outputs:
        model = model_setup(params)
        model = load_weigths_only(model, params)
        model.to(device)
        model.eval()

    valid_loader = dataloaders.get_dataloaders_test(params)

    with torch.no_grad():
        total_iou, total_maps = 0, np.array([0, 0, 0, 0, 0, 0])
        for batch_idx, (batch_images, batch_targets, images_info) in enumerate(valid_loader):
            if model_outputs:
                batch_images = batch_images.to(device)
                predictions = model(batch_images)
            else:
                n_classes = len(classes_config.training_ids)
                predictions = [torch.randn(params.batch_size, 4, anchor_config.total_anchors),
                               torch.randn(params.batch_size, n_classes, anchor_config.total_anchors)]

            for idx in range(len(batch_images)):
                non_background = batch_targets[1][idx] != 100
                all_anchor_classes = batch_targets[1][idx]
                gt_bbox = batch_targets[0][idx][non_background]
                gt_class = batch_targets[1][idx][non_background]

                iou, maps = anchor_mapping.test_anchor_mapping(
                    image=batch_images[idx], bbox_predictions=predictions[0][idx].permute(1, 0),
                    classification_predictions=predictions[1][idx].permute(1, 0),
                    gt_bbox=gt_bbox, gt_class=gt_class, image_info=images_info[idx], params=params,
                    model_outputs=model_outputs, visualize_anchors=visualize_anchors,
                    visualize_anchor_gt_pair=visualize_anchor_gt_pair, all_anchor_classes=all_anchor_classes,
                    verbose=verbose, very_verbose=very_verbose)
                total_iou += iou
                total_maps += maps

            avg = (batch_idx + 1) * params.batch_size
            print("Mean iou so far: ", total_iou / avg)
            print("Mean maps so far: ", total_maps / avg)
            if batch_idx == 10:
                return

if __name__ == '__main__':
    model_output_pipeline()
