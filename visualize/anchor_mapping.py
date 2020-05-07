import numpy as np
import random

from misc.model_output_handler import Model_output_handler

from utils.postprocessing import plot_bounding_boxes, plot_anchor_gt, nms
from utils.box_computations import jaccard, wh2corners_numpy, get_IoU, wh2corners
from utils.preprocessing import map_to_ground_truth, map_id_to_idx

from general_config.anchor_config import default_boxes, feat_size, k_list


def visualize_anchor_sets(image, anchor_grid, grid_size, k, size):
    """
    prints all anchors of a (scale, ratio) in the grid
    """
    for i in range(k):
        cur_anchors = []
        for j in range(grid_size**2):
            if random.random() > 0.75:
                cur_anchors.append(anchor_grid[i + j*k])
        cur_anchors = np.array(cur_anchors)
        plot_bounding_boxes(image=image, bounding_boxes=cur_anchors,
                            classes=np.ones(cur_anchors.shape), ground_truth=False,
                            message="Grid size: " + str(grid_size), size=size)


def visualize_all_anchor_types(image, anchors, size, sizes_ks):
    """
    visually inspect all anchors
    """
    slice_idx = 0
    for (grid_size, k) in sizes_ks:
        if k == 0:
            continue
        visualize_anchor_sets(image=image, anchor_grid=anchors[slice_idx:slice_idx+grid_size**2 * k],
                              grid_size=grid_size, k=k, size=size)
        slice_idx += grid_size**2 * k


def mapping_per_set(pos_idx, sizes_ks):
    """
    returns the number of mapped anchors from each grid
    """
    grid_maps = [0, 0, 0, 0, 0, 0]
    grid_threshold = [size**2 * k for (size, k) in sizes_ks]
    for i in range(1, len(grid_threshold)):
        grid_threshold[i] += grid_threshold[i-1]

    for idx in pos_idx:
        if idx < grid_threshold[0]:
            grid_maps[0] += 1
        elif idx < grid_threshold[1]:
            grid_maps[1] += 1
        elif idx < grid_threshold[2]:
            grid_maps[2] += 1
        elif idx < grid_threshold[3]:
            grid_maps[3] += 1
        elif idx < grid_threshold[4]:
            grid_maps[4] += 1
        else:
            grid_maps[5] += 1

    print("Anchors per grid matched this image: ", grid_maps)
    print('--------------------------------------------------------------------------------')
    print('--------------------------------------------------------------------------------')
    return np.array(grid_maps)


def mean_mapping_IOU(image, anchors, pos_idx, gt_bbox_for_matched_anchors, size, sizes_ks, visualize_anchor_gt_pair):
    """
    Checks how well anchors match ground truth bboxes
    returns the mean IoU of mapped anchors, plots pairs of anchor/gt bboxes
    """
    anchors_ltrb = wh2corners_numpy(anchors[:, :2], anchors[:, 2:])
    gt_bbox_cnr = wh2corners_numpy(
        gt_bbox_for_matched_anchors[:, :2], gt_bbox_for_matched_anchors[:, 2:])
    ious = []
    for i in range(pos_idx.shape[0]):
        cur_iou = get_IoU(anchors_ltrb[pos_idx[i]], gt_bbox_cnr[i])
        ious.append(cur_iou)
        if visualize_anchor_gt_pair:
            cur_grid = list(mapping_per_set([pos_idx[i]], sizes_ks=sizes_ks)).index(1)
            plot_anchor_gt(image, anchors[pos_idx[i]],
                           gt_bbox_for_matched_anchors[i],
                           message="Anchor/GT pair IoU: " + str(round(cur_iou, 2)) + " Grid " + str(cur_grid), size=size)

    ious = np.array(ious)
    print("Mean anchor mapping IoU for this image: ", ious.mean())
    print("Sorted: ", sorted(ious))
    print('--------------------------------------------------------------------------------------')
    return ious.mean()


def inspect_anchors(image, anchors, gt_bbox_for_matched_anchors, gt_classes_for_matched_anchors,
                    pos_idx, size, visualize_anchors, visualize_anchor_gt_pair):
    """
    thoroughly inspect anchors and mapping
    returns the mean IoU of mapped anchors and the number of mapped anchors for each grid
    """
    sizes_ks = list(zip(feat_size, k_list))
    print(sizes_ks)
    if visualize_anchors:
        visualize_all_anchor_types(image=image, anchors=anchors, size=size,
                                   sizes_ks=sizes_ks)

    iou = mean_mapping_IOU(image=image, anchors=anchors, pos_idx=pos_idx,
                           gt_bbox_for_matched_anchors=gt_bbox_for_matched_anchors,
                           size=size, sizes_ks=sizes_ks,
                           visualize_anchor_gt_pair=visualize_anchor_gt_pair)

    maps = mapping_per_set(pos_idx, sizes_ks=sizes_ks)
    return iou, maps


def test_anchor_mapping(image, bbox_predictions, classification_predictions, gt_bbox, gt_class,
                        image_info, params, model_outputs, visualize_anchors, visualize_anchor_gt_pair,
                        all_anchor_classes, verbose=False, very_verbose=False):
    """
    Args:
    image - C x H x W normalized tensor
    bbox_predictions - 4 x #anchors tensor
    classification_predictions - #classes x #anchors tensor
    gt_bbox - 4 x #anchors tensor
    gt_class - #classes x #anchors tensor
    image_info - (image_id, (width, height))
    model_outputs - flag to check model outputs or not
    visualize_anchors, visualize_anchor_gt_pair - similar flags
    """
    output_handler = Model_output_handler(params)

    anchors_ltrb = default_boxes(order="ltrb")
    anchors_xywh = default_boxes(order="xywh")

    overlaps = jaccard(wh2corners(gt_bbox[:, :2], gt_bbox[:, 2:]), anchors_ltrb)

    processed_predicted_bboxes, processed_predicted_classes, highest_confidence_for_predictions, _ = output_handler._get_sorted_predictions(
        bbox_predictions, classification_predictions, image_info)

    # map each anchor to the highest IOU obj, gt_idx - ids of mapped objects
    gt_bbox_for_matched_anchors, matched_gt_class_ids, pos_idx = map_to_ground_truth(
        overlaps, gt_bbox, gt_class, params)

    indeces_kept_by_nms = nms(wh2corners_numpy(processed_predicted_bboxes[:, :2], processed_predicted_bboxes[:, 2:]),
                              processed_predicted_classes,
                              output_handler.suppress_threshold)

    # get things in the right format
    image = output_handler._unnorm_scale_image(image)
    pos_idx = (pos_idx.cpu().numpy())
    gt_bbox = output_handler._rescale_bboxes(gt_bbox, image_info[1])
    gt_class = gt_class.cpu().numpy()
    all_anchor_classes = map_id_to_idx(all_anchor_classes).cpu().numpy()

    # get model predictions, unsorted and no nms
    raw_bbox_predictions = output_handler._convert_offsets_to_bboxes(
        bbox_predictions, image_info[1])
    raw_class_confidences = output_handler._convert_confidences_to_workable_data(
        classification_predictions)
    raw_class_indeces, _ = output_handler._get_predicted_class(raw_class_confidences)

    # rescale gt bboxes and anchors
    gt_bbox_for_matched_anchors = output_handler._rescale_bboxes(
        gt_bbox_for_matched_anchors, image_info[1])
    matched_gt_class_idxs = map_id_to_idx(matched_gt_class_ids[pos_idx]).cpu().numpy()
    anchors_xywh = output_handler._rescale_bboxes(anchors_xywh, image_info[1])

    if model_outputs:
        test(raw_bbox=raw_bbox_predictions, raw_class_confidences=classification_predictions, raw_class_indeces=raw_class_indeces,
             gt_bbox=gt_bbox, gt_class=gt_class,
             pred_bbox=processed_predicted_bboxes, pred_class=processed_predicted_classes,
             highest_confidence_for_predictions=highest_confidence_for_predictions,
             indeces_kept_by_nms=indeces_kept_by_nms,
             pos_idx=pos_idx,
             size=image_info[1],
             image=image,
             anchors=anchors_xywh,
             gt_bbox_for_matched_anchors=gt_bbox_for_matched_anchors,
             matched_gt_class_idxs=matched_gt_class_idxs,
             all_anchor_classes=all_anchor_classes,
             verbose=verbose,
             very_verbose=very_verbose)

    return inspect_anchors(image=image, anchors=anchors_xywh, gt_bbox_for_matched_anchors=gt_bbox_for_matched_anchors,
                           gt_classes_for_matched_anchors=matched_gt_class_idxs, pos_idx=pos_idx, size=image_info[1],
                           visualize_anchors=visualize_anchors, visualize_anchor_gt_pair=visualize_anchor_gt_pair)


def test(raw_bbox=None, raw_class_confidences=None, raw_class_indeces=None,
         gt_bbox=None, gt_class=None,
         pred_bbox=None, pred_class=None,
         highest_confidence_for_predictions=None,
         indeces_kept_by_nms=None,
         pos_idx=None, size=(320, 320),
         image=None, anchors=None,
         gt_bbox_for_matched_anchors=None,
         matched_gt_class_idxs=None,
         all_anchor_classes=None,
         verbose=False,
         very_verbose=False):
    """
    what we have:
    - raw bbox and class - all the model predictions (not filtered, not sorted, no nms)
    - gt_bbox and class - the ground truth for the image
    - pred_bbox and pred_class - sorted model predictions by confidence higher than a threshold
    - highest_confidence_for_predictions - what the maximum confidence for the respective prediction is
    - high_confidence_indeces - indeces of the highest confidence predictions (slice raw_bbox by this and get pred_bbox)
    - pos_idx - indeces of anchors (predictions) that mapped by IOU threshold (matching phase)
    - size - dimensions of image
    - image - actual input image
    - verbose - whether or not to print details
    - very_verbose - print every anchor with its respective prediction, one by one
    """
    matched_anchors = anchors[pos_idx]
    matched_bbox = raw_bbox[pos_idx]
    matched_conf = raw_class_confidences[pos_idx]
    matched_indeces = raw_class_indeces[pos_idx]

    # add the classes array for each bbox array
    plot_bounding_boxes(image=image, bounding_boxes=gt_bbox, classes=gt_class,
                        ground_truth=True, message="Ground truth", size=size)

    # plot the ground truth bbox for each matched anchor, it is possible in some cases that there is no anchor for a gt bbox
    plot_bounding_boxes(image=image, bounding_boxes=gt_bbox_for_matched_anchors, classes=matched_gt_class_idxs,
                        ground_truth=True, message="Matched Ground truth", size=size)

    # plot matched anchors
    plot_bounding_boxes(image=image, bounding_boxes=matched_anchors,
                        classes=matched_gt_class_idxs, ground_truth=False, message="Anchors", size=size)

    # plot model predictions from the matched anchors, ideally, these should also have the highest confidence
    plot_bounding_boxes(image=image, bounding_boxes=matched_bbox, classes=matched_indeces,
                        ground_truth=False, message="Cheated Predictions", size=size)

    # plot model predictions before applying NMS
    plot_bounding_boxes(image=image, bounding_boxes=pred_bbox, classes=pred_class,
                        ground_truth=False, message="Pre NMS Predictions", size=size)

    post_nms_predictions = pred_bbox[indeces_kept_by_nms]
    post_nms_classes = pred_class[indeces_kept_by_nms]
    plot_bounding_boxes(image=image, bounding_boxes=post_nms_predictions,
                        classes=post_nms_classes, ground_truth=False, message="Post NMS Predictions", size=size)
    if verbose:
        print("GT BBOXES: ", gt_bbox, gt_bbox.shape)

        print("MATCHED GT BBOXES: ", gt_bbox_for_matched_anchors, gt_bbox_for_matched_anchors.shape)

        print("Matched ANCHORS ", matched_anchors, matched_anchors.shape)

        print("Matched Pred BBOXES: (Cheated preditctions) ", matched_bbox, matched_bbox.shape)
        print('CONFIDENCES FOR PREDICTED BBOXES that matched anchors: ', matched_conf)

        print("THIS IS PRED BBOX KEPT BY CONFIDENCE", pred_bbox, pred_bbox.shape)
        print("These are confidences for model outputs: ", highest_confidence_for_predictions)

        print("THIS IS POST NMS PREDICTIONS", post_nms_predictions,
              post_nms_predictions.shape)

        plot_pred_anchor(image=image, anchors=matched_anchors, pred_bbox=matched_bbox,
                         anchor_classes=matched_gt_class_idxs, pred_classes=matched_indeces, size=size)

    if very_verbose:
        plot_pred_anchor(image=image, anchors=anchors, pred_bbox=raw_bbox,
                         anchor_classes=all_anchor_classes, pred_classes=raw_class_indeces, size=size)


def plot_pred_anchor(image, anchors, pred_bbox, anchor_classes, pred_classes, size):
    for cur_anchor_bbox, cur_pred_bbox, cur_anchor_idx, cur_pred_idx in zip(anchors, pred_bbox, anchor_classes, pred_classes):
        plot_bounding_boxes(image=image, bounding_boxes=cur_anchor_bbox,
                            classes=cur_anchor_idx, ground_truth=False, message="ANCHOR", size=size)
        plot_bounding_boxes(image=image, bounding_boxes=cur_pred_bbox, classes=cur_pred_idx,
                            ground_truth=False, message="PRED FROM ANCHOR", size=size)
        print("Current class of anchor: ", cur_anchor_idx)
        print("Current predicted class: ", cur_pred_idx)
        # print('Confidence for this pair of anchor/pred: ',
        #       raw_class_confidences[i], size)
