import numpy as np
import cv2

import json
from general_config import classes_config, path_config


from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


def nms(bounding_boxes, predicted_classes, threshold=0.5):
    """
    args:
        bounding_boxes: nr_bboxes x 4 sorted by confidence
        threshold: bboxes with IoU above threshold will be removed

    returns:
        final_model_predictions: nr_bboxes x 4

    bounding_boxes MUST be sorted
    """
    bounding_boxes = bounding_boxes[:200]
    predicted_classes = predicted_classes[:200]

    indices = np.array(range(bounding_boxes.shape[0]))
    final_model_predictions = []
    while indices.shape[0] != 0:
        prediction = bounding_boxes[indices[0]]
        final_model_predictions.append(indices[0])

        to_keep = []
        for index in range(indices.shape[0]):
            IoU = get_IoU(prediction, bounding_boxes[indices[index]])

            if IoU < threshold or (predicted_classes[indices[0]] != predicted_classes[indices[index]]):
                to_keep.append(index)

        indices = indices[to_keep]

    return final_model_predictions


def after_nms(prediction_bboxes, predicted_confidences):
    """
    return final model preditctions
    """
    kept_after_nms = nms(prediction_bboxes)

    post_nms_bboxes = prediction_bboxes[kept_after_nms]
    post_nms_confidences = predicted_confidences[kept_after_nms]

    return post_nms_bboxes, post_nms_confidences


def get_intersection(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    return np.array([x1, y1, x2, y2])


def get_bbox_area(bbox):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    if width < 0 or height < 0:
        return 0

    return width*height


def get_IoU(bbox1, bbox2):
    bbox1_x1, bbox1_y1, bbox1_x2, bbox1_y2 = bbox1
    bbox2_x1, bbox2_y1, bbox2_x2, bbox2_y2 = bbox2

    intersection = get_intersection(bbox1, bbox2)
    intersection_area = get_bbox_area(intersection)

    union_area = get_bbox_area(bbox1) + get_bbox_area(bbox2) - intersection_area

    if union_area == 0:
        return -1

    return intersection_area/union_area


def plot_anchor_gt(image, anchor, gt, cur_class, message="DA_MA", size=(320, 320)):
    image = image.transpose(1, 2, 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(size[0], size[1]))

    color_anchor = (0, 0, 255)
    cv2.rectangle(image, (int(anchor[0]-anchor[2]/2), int(anchor[1]-anchor[3]/2)),
                  (int(anchor[0] + anchor[2]/2), int(anchor[1] + anchor[3]/2)), color_anchor, 2)

    # gt_id_2_color = {1: (200, 200, 0), 3: (150, 250, 150)}
    color_gt = (200, 200, 0)
    gt_id_2_color = {1: (200, 200, 0), 3: (150, 250, 150)}
    color_gt = gt_id_2_color.get(cur_class, color_gt)
    cv2.rectangle(image, (int(gt[0]-gt[2]/2), int(gt[1]-gt[3]/2)),
                  (int(gt[0] + gt[2]/2), int(gt[1] + gt[3]/2)), color_gt, 2)

    cv2.imshow(message, image)
    cv2.waitKey(0)


def plot_bounding_boxes(image, bounding_boxes, classes, bbox_type="pred", message='no_message', size=(500, 500)):
    """
    Plots an array of bounding_boxes with their respective color, returns the modified image
    """
    image = image.transpose(1, 2, 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(size[0], size[1]))

    # light blue gt is human, light green is vehicle
    gt_id_2_color = {1: (200, 200, 0), 3: (150, 250, 150)}
    # blue prediction is human, green is vehicle
    pred_id_2_color = {1: (255, 0, 0), 3: (0, 255, 0)}
    # anchors are not class aware, they are just red
    anchor_id_2_color = {1: (0, 0, 255), 3: (0, 0, 255)}
    if bbox_type == "pred":
        id_2_color = pred_id_2_color
    elif bbox_type == "gt":
        id_2_color = gt_id_2_color
    else:
        id_2_color = anchor_id_2_color

    if len(bounding_boxes.shape) == 1:
        bounding_boxes = bounding_boxes.reshape(1, -1)
    classes = classes.reshape(-1)

    for (startX, startY, width, height), pred_class in zip(bounding_boxes, classes):
        color = id_2_color.get(pred_class, (0, 255, 0))
        cv2.rectangle(image, (int(startX-width/2), int(startY-height/2)),
                      (int(startX+width/2), int(startY+height/2)), color, 2)

    # display the image
    cv2.imshow(message, image)
    cv2.waitKey(0)

    return image


def prepare_outputs_for_COCOeval(output, image_info, prediction_annotations, prediction_id, output_handler):
    batch_size = output[0].shape[0]

    for i in range(batch_size):

        image_id = image_info[i][0]

        complete_outputs = output_handler.process_outputs(
            output[0][i], output[1][i], image_info[i])

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

    graundtrutu = COCO(path_config.val_annotations_path)
    predictile = graundtrutu.loadRes('fisierul.json')

    cocoevalu = COCOeval(graundtrutu, predictile, iouType='bbox')

    # cocoevalu.params.catIds = [1, 3]

    cocoevalu.evaluate()
    cocoevalu.accumulate()
    cocoevalu.summarize()

    return cocoevalu.stats[0]
