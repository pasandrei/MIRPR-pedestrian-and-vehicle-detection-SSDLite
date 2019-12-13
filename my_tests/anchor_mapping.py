import torch
from torch import nn
import numpy as np
# import torch.nn.functional as F

from train.helpers import *
from misc import postprocessing
from misc.utils import *
from misc.model_output_handler import *


def visualize_anchor_sets(image, anchor_grid, grid_size, k, size, gt_bbox_for_matched_anchors):
    """
    prints all anchors of a (scale, ratio) in the grid
    """
    print(anchor_grid.shape)
    # for i in range(k):
    #     cur_anchors = []
    #     for j in range(grid_size, 2*grid_size):
    #         cur_anchors.append(anchor_grid[i + j*k])
    #     plot_bounding_boxes(image, np.array(cur_anchors), "Set " + str(i), size)
    only = gt_bbox_for_matched_anchors[0]
    print("This is a gt bbox: ", only)
    for anchor in anchor_grid:
        print(get_IoU(only, anchor))
        plot_bounding_boxes(image, anchor, size=size)


def visualize_all_anchor_types(image, anchors, size, gt_bbox_for_matched_anchors):
    """
    currently there's 10x10x12 + (5x5 + 3x3 + 2x2 + 1x1) * 20 anchors
    want to check these
    """
    slice_idx = 10*10*12
    # _10x10 = anchors[:slice_idx]
    # visualize_anchor_sets(image=image, anchor_grid=_10x10, grid_size=10, k=12, size=size)
    # _5x5 = anchors[slice_idx:slice_idx + 5*5*20]
    # slice_idx += 5*5*20
    # visualize_anchor_sets(image=image, anchor_grid=_5x5, grid_size=5, k=20, size=size)
    # _3x3 = anchors[slice_idx:slice_idx + 3*3*20]
    # slice_idx += 3*3*20
    # visualize_anchor_sets(image=image, anchor_grid=_3x3, grid_size=3, k=20, size=size)
    # _2x2 = anchors[slice_idx:slice_idx + 2*2*20]
    # slice_idx += 2*2*20
    # visualize_anchor_sets(image=image, anchor_grid=_2x2, grid_size=2, k=20, size=size)
    # _1x1 = anchors[slice_idx:slice_idx + 1*1*20]
    # visualize_anchor_sets(image=image, anchor_grid=_1x1, grid_size=1, k=20, size=size)
    _1x1 = anchors[-50:]
    visualize_anchor_sets(image=image, anchor_grid=_1x1, grid_size=1,
                          k=50, size=size, gt_bbox_for_matched_anchors=gt_bbox_for_matched_anchors)


def inspect_anchors(image, anchors, gt_bbox_for_matched_anchors, pos_idx, size):
    """
    thoroughly inspect anchors and mapping
    """
    # visualize_all_anchor_types(image=image, anchors=anchors, size=size,
    #                            gt_bbox_for_matched_anchors=gt_bbox_for_matched_anchors)

    # check anchor and gt match
    for i in range(pos_idx.shape[0]):
        plot_anchor_gt(image, anchors[pos_idx[i]],
                       gt_bbox_for_matched_anchors[i], message=str(i), size=size)


def mean_mapping_IOU(anchors, pos_idx, gt_bbox_for_matched_anchors):
    ious = []
    for i in range(pos_idx.shape[0]):
        ious.append(get_IoU(anchors[pos_idx[i]], gt_bbox_for_matched_anchors[i]))

    ious = np.array(ious)
    print("Mean anchor mapping IoU for this image: ", ious.mean())
    print("Sorted: ", sorted(ious))
    print('--------------------------------------------------------------------------------------')
    return ious.mean()


def mapping_per_set(pos_idx):
    # grid_maps = [0, 0, 0, 0, 0]
    # thresh_10 = 10*10*12
    # thresh_5 = thresh_10 + 5*5*20
    # thresh_3 = thresh_5 + 3*3*20
    # thresh_2 = thresh_3 + 2*2*20
    # for idx in pos_idx:
    #     if idx < thresh_10:
    #         grid_maps[0] += 1
    #     elif idx < thresh_5:
    #         grid_maps[1] += 1
    #     elif idx < thresh_3:
    #         grid_maps[2] += 1
    #     elif idx < thresh_2:
    #         grid_maps[3] += 1
    #     else:
    #         grid_maps[4] += 1

    grid_maps = [0, 0, 0, 0, 0, 0]
    thresh_20 = 20*20*12
    thresh_10 = thresh_20 + 10*10*20
    thresh_5 = thresh_10 + 5*5*30
    thresh_3 = thresh_5 + 3*3*40
    thresh_2 = thresh_3 + 2*2*50
    for idx in pos_idx:
        if idx < thresh_20:
            grid_maps[0] += 1
        elif idx < thresh_10:
            grid_maps[1] += 1
        elif idx < thresh_5:
            grid_maps[2] += 1
        elif idx < thresh_3:
            grid_maps[3] += 1
        elif idx < thresh_2:
            grid_maps[4] += 1
        else:
            grid_maps[5] += 1

    print("Anchors per grid matched this image: ", grid_maps)
    print('--------------------------------------------------------------------------------')
    return np.array(grid_maps)


def test_anchor_mapping(image, bbox_predictions, classification_predictions, gt_bbox, gt_class, image_info):
    """
    Args: all input is required per image

    computes:
        - image upscaled and unnormalized as numpy array
        - predicted bboxes higher than a threshold, sorted by predicted confidence, at right scale
        - gt bboxes for the image, at right scale
    """
    output_handler = Model_output_handler()
    bbox_predictions = bbox_predictions.cpu()

    overlaps = jaccard(gt_bbox, output_handler.corner_anchors)

    prediction_bboxes, predicted_classes, highest_confidence_for_predictions, high_confidence_indeces = output_handler._get_sorted_predictions(
        bbox_predictions, classification_predictions, image_info)

    print("THESE BBOXES SHOULD LOOK OK", prediction_bboxes)

    # map each anchor to the highest IOU obj, gt_idx - ids of mapped objects
    gt_bbox_for_matched_anchors, _, pos_idx = map_to_ground_truth(
        overlaps, gt_bbox, gt_class)

    # inexplicable bug
    print('nms on YOOOO postpros')
    indeces_kept_by_nms = postprocessing.nms(prediction_bboxes, predicted_classes,
                                             output_handler.suppress_threshold)
    print('indices kept by nms')

    image = output_handler._unnorm_scale_image(image)
    pos_idx = (pos_idx.cpu().numpy())
    gt_bbox = output_handler._rescale_bboxes(gt_bbox, image_info[1])
    # bbox_predictions = output_handler._convert_bboxes_to_workable_data(
    #     bbox_predictions, image_info[1])

    iou = mean_mapping_IOU(output_handler._rescale_bboxes(output_handler.corner_anchors,
                                                          image_info[1]), pos_idx, output_handler._rescale_bboxes(gt_bbox_for_matched_anchors, image_info[1]))
    maps = mapping_per_set(pos_idx)

    test(raw_bbox=output_handler._convert_bboxes_to_workable_data(bbox_predictions, image_info[1]), image=image, anchors=output_handler._rescale_bboxes(output_handler.corner_anchors, image_info[1]),
         pred_bbox=prediction_bboxes, highest_confidence_for_predictions=highest_confidence_for_predictions,
         gt_bbox=gt_bbox, pos_idx=pos_idx, high_confidence_indeces=high_confidence_indeces, size=image_info[1], indeces_kept_by_nms=indeces_kept_by_nms)

    # inspect_anchors(image=image, anchors=output_handler._rescale_bboxes(
    #     output_handler.corner_anchors, image_info[1]),
    #     gt_bbox_for_matched_anchors=output_handler._rescale_bboxes(gt_bbox_for_matched_anchors, image_info[1]), pos_idx=pos_idx, size=image_info[1])

    return iou, maps


def test(raw_bbox=None, image=None, anchors=None, pred_bbox=None, highest_confidence_for_predictions=None,
         gt_bbox=None, pos_idx=None, high_confidence_indeces=None, size=(320, 320), indeces_kept_by_nms=None):
    '''
    what we have:
                  - raw_bbox - all model bbox predictions
                  - image: C x H x W tensor
                  # anchors x 4 tensor, scaled in [0, 1]: the set of predifined bounding boxes
                  - anchors:
                  # anchors x 4 tensor, in [0, 1] scale: sorted by confidence higher than threshold
                  - pred_bbox:
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
    print("Matched Pred BBOXES: ", matched_bbox, matched_bbox.shape)
    # for i in range(len(matched_anchors)):
    #     cur_anchor_bbox = matched_anchors[i]
    #     cur_pred_bbox = matched_bbox[i]
    #     plot_bounding_boxes(image, cur_anchor_bbox, "ANCHOR", size)
    #     plot_bounding_boxes(image, cur_pred_bbox, "PRED FROM ANCHOR", size)
    #     print('Confidence for this pair of anchor/pred: ',
    #           highest_confidence_for_predictions[i], size)

    print('CONFIDENCES FOR PREDICTED BBOXES: ', highest_confidence_for_predictions)
    plot_bounding_boxes(image, matched_bbox, "PREDICTED (CHEATED) BY THE NETWORK", size)
    # plot_bounding_boxes(image, matched_anchors, "MATCHED ANCHORS", size)

    print("THIS IS PRED BBOX KEPT BY CONFIDENCE", pred_bbox,
          pred_bbox.shape)
    plot_bounding_boxes(image, pred_bbox, "ACTUAL MODEL OUTPUTS", size)
    post_nms_predictions = pred_bbox[indeces_kept_by_nms]
    plot_bounding_boxes(
        image, post_nms_predictions, 'Post NMS predictions', size)
    print("THIS IS POST NMS PREDICTIONS", post_nms_predictions,
          post_nms_predictions.shape)
