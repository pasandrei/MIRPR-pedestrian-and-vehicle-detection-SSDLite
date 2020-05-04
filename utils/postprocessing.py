import numpy as np
import cv2

import json
from general_config import classes_config, constants
from utils.box_computations import get_IoU


from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


def remove_overlapping_bboxes(current_class_indeces, bounding_boxes, thresold):
    """
    Args:
    current_class_indeces: ndarray of current class indeces, sorted decreasingly by confidence
    bounding_boxes: ndarray of predicted bboxes
    """
    kept = []
    eliminated = {}
    for i in range(len(current_class_indeces)):
        if i in eliminated:
            continue
        else:
            kept.append(i)
        # those that intersect with i are eliminated, since i is more confident
        for j in range(i+1, len(current_class_indeces)):
            IoU = get_IoU(bounding_boxes[i], bounding_boxes[j])
            if IoU >= thresold:
                eliminated[j] = 1
    return kept


def nms(bounding_boxes, predicted_classes, threshold=0.5):
    """
    args:
        bounding_boxes: nr_bboxes x 4 sorted by confidence
        predicted_classes: classes predicted by the model
        threshold: bboxes with IoU above threshold will be removed

    returns:
        final_model_predictions: indices of kept bboxes

    bounding_boxes are sorted decreasingly by confidence
    """
    # keep top 100 predictions
    bounding_boxes = bounding_boxes[:100]
    predicted_classes = predicted_classes[:100]

    final_model_predictions = []
    for id in np.unique(predicted_classes):
        # get indeces of current class
        current_class_indeces = np.argwhere(predicted_classes == id)
        kept_indeces = remove_overlapping_bboxes(current_class_indeces, bounding_boxes, threshold)
        final_model_predictions.extend(kept_indeces)

    return final_model_predictions


def after_nms(prediction_bboxes, predicted_confidences):
    """
    return final model preditctions
    """
    kept_after_nms = nms(prediction_bboxes)

    post_nms_bboxes = prediction_bboxes[kept_after_nms]
    post_nms_confidences = predicted_confidences[kept_after_nms]

    return post_nms_bboxes, post_nms_confidences


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
            bbox = [int(x) for x in complete_outputs[index][:4]]

            prediction_id += 1
            prediction_annotations.append(
                {"image_id": image_id, "bbox": bbox,
                 "score": float(complete_outputs[index][5]),
                 "category_id": int(classes_config.idx_training_ids2[complete_outputs[index][4]]), "id": prediction_id})

    return prediction_annotations, prediction_id


def evaluate_on_COCO_metrics(prediction_annotations):
    with open("fisierul.json", 'w') as f:
        json.dump(prediction_annotations, f)

    ground_truth = COCO(constants.val_annotations_path)
    predictions = ground_truth.loadRes('fisierul.json')

    cocoevalu = COCOeval(ground_truth, predictions, iouType='bbox')

    # cocoevalu.params.catIds = [1, 3]

    cocoevalu.evaluate()
    cocoevalu.accumulate()
    cocoevalu.summarize()

    return cocoevalu.stats[0]
