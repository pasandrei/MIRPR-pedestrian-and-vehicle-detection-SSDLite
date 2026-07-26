import numpy as np
import cv2
import torch
import torchvision

import json
from general_config import classes_config, constants, general_config


from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


def nms(bounding_boxes, predicted_classes, threshold=0.5):
    """
    args:
        bounding_boxes: nr_bboxes x 4 numpy array in corner format [x1,y1,x2,y2], sorted by confidence
        predicted_classes: numpy array of classes predicted by the model
        threshold: bboxes with IoU above threshold will be removed

    returns:
        final_model_predictions: indices of kept bboxes
    """
    bounding_boxes = bounding_boxes[:1000]
    predicted_classes = predicted_classes[:1000]

    n = bounding_boxes.shape[0]
    if n == 0:
        return []

    boxes_t = torch.from_numpy(bounding_boxes).float()
    # scores derived from sort order (already sorted by decreasing confidence)
    scores = torch.arange(n, 0, -1).float()

    if general_config.agnostic_nms:
        keep = torchvision.ops.nms(boxes_t, scores, threshold)
        return keep.numpy().tolist()
    else:
        classes_t = torch.from_numpy(predicted_classes.reshape(-1)).float()
        keep = torchvision.ops.batched_nms(boxes_t, scores, classes_t, threshold)
        return keep.numpy().tolist()


def plot_anchor_gt(image, anchor, gt, message="no_message", size=(320, 320)):
    """
    Plots a ground truth bbox with an anchor that mapped to it
    """
    image = image.transpose(1, 2, 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(size[0], size[1]))

    color_anchor = (0, 0, 255)
    cv2.rectangle(image, (int(anchor[0]-anchor[2]/2), int(anchor[1]-anchor[3]/2)),
                  (int(anchor[0] + anchor[2]/2), int(anchor[1] + anchor[3]/2)), color_anchor, 2)

    color_gt = (255, 0, 0)
    cv2.rectangle(image, (int(gt[0]-gt[2]/2), int(gt[1]-gt[3]/2)),
                  (int(gt[0] + gt[2]/2), int(gt[1] + gt[3]/2)), color_gt, 2)

    cv2.imshow(message, image)
    cv2.waitKey(0)


def plot_bounding_boxes(image, bounding_boxes, classes, ground_truth=False, message='no_message', size=(320, 320)):
    """
    Plots an array of bounding_boxes with their respective color, returns the modified image
    red - ground truth
    green - person prediction
    blue - vehicle prediction
    gray - any other class
    """
    image = image.transpose(1, 2, 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(size[0], size[1]))

    if len(bounding_boxes.shape) == 1:
        bounding_boxes = bounding_boxes.reshape(1, -1)
    classes = classes.reshape(-1)

    for (startX, startY, width, height), pred_class in zip(bounding_boxes, classes):
        color = (255, 0, 0) if ground_truth else classes_config.complete_map.get(
            pred_class, (170, 170, 170))
        cv2.rectangle(image, (int(startX-width/2), int(startY-height/2)),
                      (int(startX+width/2), int(startY+height/2)), color, 2)

    # display the image
    cv2.imshow(message, image)
    cv2.waitKey(0)

    return image


def prepare_outputs_for_COCOeval(output, image_info, prediction_annotations, prediction_id, output_handler):
    """
    convert raw model outputs to format required by COCO evaluation
    """
    batch_size = output[0].shape[0]

    for i in range(batch_size):

        image_id = image_info[i][0]

        pred_bbox = output[0].permute(0, 2, 1)
        pred_class = output[1].permute(0, 2, 1)

        complete_outputs = output_handler.process_outputs(
            pred_bbox[i], pred_class[i], image_info[i])

        for index in range(complete_outputs.shape[0]):
            bbox = [round(float(x), 2) for x in complete_outputs[index][:4]]

            prediction_id += 1
            prediction_annotations.append(
                {"image_id": image_id, "bbox": bbox,
                 "score": float(complete_outputs[index][5]),
                 "category_id": int(classes_config.idx_training_ids2[complete_outputs[index][4]]), "id": prediction_id})

    return prediction_annotations, prediction_id


def evaluate_on_COCO_metrics(prediction_annotations):
    if len(prediction_annotations) == 0:
        print("No predictions generated, mAP = 0")
        return 0.0

    with open("fisierul.json", 'w') as f:
        json.dump(prediction_annotations, f)

    ground_truth = COCO(constants.val_annotations_path)
    predictions = ground_truth.loadRes('fisierul.json')

    cocoevalu = COCOeval(ground_truth, predictions, iouType='bbox')

    # cocoevalu.params.catIds = classes_config.eval_cat_ids

    cocoevalu.evaluate()
    cocoevalu.accumulate()
    cocoevalu.summarize()

    return cocoevalu.stats[0]


def postprocess_until_nms(output_handler, pred_boxes, pred_confs, img_size=(300, 300)):
    """
    Processes immediate model outputs to get data for nms

    Args:
    pred_boxes: #anchors x 4 tensor scaled in range [0,1]
    pred_confs: #anchors x n_classes tensor of confidences

    Returns:
    ndarrays
    """
    # get numpy arrays of rescaled boxes and confidences minus background
    pred_boxes, pred_confs = output_handler._convert_output_to_workable_data(
        pred_boxes, pred_confs, img_size)

    # cut predictions that are below confidence threshold
    pred_boxes, pred_confs = output_handler._predictions_over_threshold(pred_boxes,
                                                                        pred_confs)

    # get predicted classes and respective confidences
    pred_classes, highest_confidence_for_predictions = output_handler._get_predicted_class(
        pred_confs)

    # sort decreasingly
    permutation = (-highest_confidence_for_predictions).argsort()
    pred_boxes = pred_boxes[permutation]
    pred_classes = pred_classes[permutation]

    return pred_boxes, pred_classes


def clip_boxes(boxes, width, height):
    """
    boxes: ndarray of (x1,y1,x2,y2) boxes
    """
    boxes[:, 0] = np.clip(boxes[:, 0], 0, width-1)
    boxes[:, 2] = np.clip(boxes[:, 2], 0, width-1)
    boxes[:, 1] = np.clip(boxes[:, 1], 0, height-1)
    boxes[:, 3] = np.clip(boxes[:, 3], 0, height-1)
