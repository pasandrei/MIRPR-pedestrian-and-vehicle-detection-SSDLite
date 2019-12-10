import numpy as np
import cv2
from train.helpers import activations_to_bboxes

# postprocess


def convert_bboxes_to_workable_data(prediction_bboxes, anchors, grid_sizes, size):
    height, width = size
    prediction_bboxes = activations_to_bboxes(prediction_bboxes, anchors, grid_sizes)
    prediction_bboxes[:, 0] *= height
    prediction_bboxes[:, 2] *= height
    prediction_bboxes[:, 1] *= width
    prediction_bboxes[:, 3] *= width
    prediction_bboxes = (prediction_bboxes.cpu().numpy()).astype(int)

    return prediction_bboxes


def convert_confidences_to_workable_data(prediction_confidences):
    return prediction_confidences.sigmoid().cpu().numpy()


def convert_output_to_workable_data(model_output_bboxes, model_output_confidences, anchors, grid_sizes, size):
    prediction_bboxes = convert_bboxes_to_workable_data(
        model_output_bboxes, anchors, grid_sizes, size)
    prediction_confidences = convert_confidences_to_workable_data(model_output_confidences)

    return prediction_bboxes, prediction_confidences


def nms(bounding_boxes, threshold=0.5):
    """
    args:
        bounding_boxes: nr_bboxes x 4 sorted by confidence
        threshold: bboxes with IoU above threshold will be removed

    returns:
        final_model_predictions: nr_bboxes x 4

    bounding_boxes MUST be sorted
    """

    indices = np.array(range(bounding_boxes.shape[0]))
    final_model_predictions = []
    while indices.shape[0] != 0:
        prediction = bounding_boxes[indices[0]]
        final_model_predictions.append(indices[0])

        to_keep = []
        for index in range(indices.shape[0]):
            IoU = get_IoU(prediction, bounding_boxes[indices[index]])

            if IoU < threshold:
                to_keep.append(index)

        indices = indices[to_keep]

    return final_model_predictions


def corners_to_wh(prediction_bboxes):
    for index in range(prediction_bboxes.shape[0]):
        prediction_bboxes[index][0], prediction_bboxes[index][1] = prediction_bboxes[index][1], prediction_bboxes[index][0]
        prediction_bboxes[index][2], prediction_bboxes[index][3] = prediction_bboxes[index][3], prediction_bboxes[index][2]

        width = prediction_bboxes[index][2] - prediction_bboxes[index][0]
        height = prediction_bboxes[index][3] - prediction_bboxes[index][1]

        prediction_bboxes[index][2] = width
        prediction_bboxes[index][3] = height

    return prediction_bboxes


def get_predicted_class(prediction_confidences):
    predicted_classes = np.argmax(prediction_confidences, axis=1)
    highest_confidence_for_predictions = np.amax(
        prediction_confidences, axis=1)

    return predicted_classes, highest_confidence_for_predictions


def predictions_over_threshold(prediction_bboxes, predicted_confidences, threshold=0.25):
    keep_indices = []

    for index, one_hot_pred in enumerate(predicted_confidences):
        max_conf = np.amax(one_hot_pred)
        if max_conf > threshold:
            keep_indices.append(index)

    prediction_bboxes = prediction_bboxes[keep_indices]
    predicted_confidences = predicted_confidences[keep_indices]

    return prediction_bboxes, predicted_confidences


def after_nms(prediction_bboxes, predicted_confidences):
    kept_after_nms = nms(prediction_bboxes)

    post_nms_bboxes = prediction_bboxes[kept_after_nms]
    post_nms_confidences = predicted_confidences[kept_after_nms]

    return post_nms_bboxes, post_nms_confidences

#  postprocess


def plot_bounding_boxes(image, bounding_boxes, message='no_message', ok=0):
    # loop over the bounding boxes for each image and draw them
    image = image.transpose(1, 2, 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    color = (0, 255, 0) if ok else (0, 0, 255)
    if len(bounding_boxes.shape) == 1:
        cv2.rectangle(image, (bounding_boxes[1], bounding_boxes[0]),
                      (bounding_boxes[3], bounding_boxes[2]), color, 2)
    else:
        for (startX, startY, endX, endY) in bounding_boxes:
            cv2.rectangle(image, (startY, startX), (endY, endX), color, 2)

    # display the image
    cv2.imshow(message, image)
    cv2.waitKey(0)


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
