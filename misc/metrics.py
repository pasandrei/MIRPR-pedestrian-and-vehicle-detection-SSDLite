import numpy as np


def get_predicted_class(prediction_confidences):
    predicted_classes = np.argmax(prediction_confidences, axis=1)
    highest_confidence_for_predictions = np.amax(
        prediction_confidences, axis=1)

    return predicted_classes, highest_confidence_for_predictions


def sort_predictions_by_confidence(prediction_bboxes, predicted_classes, highest_confidence_for_prediction):
    permutation = (-highest_confidence_for_prediction).argsort()

    prediction_bboxes = prediction_bboxes[permutation]
    predicted_classes = predicted_classes[permutation]
    highest_confidence_for_prediction = highest_confidence_for_prediction[permutation]

    return prediction_bboxes, predicted_classes, highest_confidence_for_prediction


def get_intersection(bbox1, bbox2):
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    return np.array([x1, y1, x2, y2])


def get_bbox_area(bbox):
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]

    return width*height


def get_IoU(bbox1, bbox2):
    bbox1_x1, bbox1_y1, bbox1_x2, bbox1_y2 = bbox1
    bbox2_x1, bbox2_y1, bbox2_x2, bbox2_y2 = bbox2

    intersection = get_intersection(bbox1, bbox2)
    intersection_area = get_bbox_area(intersection)

    union_area = get_bbox_area(bbox1) + get_bbox_area(bbox2) - intersection_area

    return intersection_area/union_area


def help_calculate_AP(gt_bboxes, gt_classes, prediction_bboxes, prediction_confidences, required_IoU):
    """
    IN:
        gt_bboxes: [[x1, y1, x2, y2] ...] #obj x 4
        gt_classes: [bbox1_class bbox2_class ..] #obj x 1
        prediction_bboxes: [[x1, y1, x2, y2] ... ] #post_nms_bboxes x 4
        prediction_confidences: [[class1_confidence class2_confidence] ...] #post_nms_predictions x 2

        prediction_confidences is asssumed to have 2 columns (one for each class)
    """

    predicted_classes, prediction_confidences = get_predicted_class(prediction_confidences)

    prediction_bboxes, predicted_classes, highest_confidence_for_prediction = sort_predictions_by_confidence(
        prediction_bboxes, predicted_classes, prediction_confidences)

    true_positives = 0
    false_positives = 0

    for index_prediction in range(prediction_bboxes.shape[0]):
        prediction_bbox = prediction_bboxes[index_prediction]
        predicted_class = predicted_classes[index_prediction]

        ok = 0
        for index_gt in range(gt_bboxes.shape[0]):
            gt_bbox = gt_bboxes[index_gt]
            gt_class = gt_classes[index_gt]

            if predicted_class != gt_class:
                continue

            current_IoU = get_IoU(prediction_bbox, gt_bbox)

            if current_IoU >= required_IoU:
                ok = 1

        true_positives += ok
        false_positives += 1-ok

    return true_positives/(true_positives+false_positives)


def calculate_AP(model_output, label, required_IoU=0.5):
    batch_size = model_output[0].shape[0]

    ap = 0
    for i in range(batch_size):
        prediction_bboxes = (model_output[0][i].cpu().numpy() * 320).astype(int)
        prediction_confidences = model_output[1][i].cpu().numpy()

        gt_bboxes = (label[0][i].cpu().numpy() * 320).astype(int)
        gt_classes = label[1][i].cpu().numpy()

        ap += help_calculate_AP(gt_bboxes, gt_classes, prediction_bboxes,
                                prediction_confidences, required_IoU)

    return ap
