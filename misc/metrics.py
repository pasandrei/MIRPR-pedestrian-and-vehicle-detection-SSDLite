def help_calculate_AP(gt_bboxes, prediction_bboxes, IoU):
    """
    IN:
        gt_bboxes: [[[x1, y1, x2, y2] class] ... ]
        prediction_bboxes: [[[x1, y1, x2, y2] class confidence] ... ]
    """

    def sort_by_confidence(prediction_bbox):
        return prediction_bbox[2]

    prediction_bboxes.sort(key=sort_by_confidence)

    true_positives = 0
    false_positives = 0

    for prediction_bbox in prediction_bboxes:
        ok = 0
        for gt_bbox in gt_bboxes:
            if prediction_bbox[1] != gt_bbox[1]:
                continue

            current_IoU = jaccard(prediction_bbox[0], gt_bbox[0])

            if current_IoU >= IoU:
                ok = 1

        true_positives += ok
        false_positives += 1-ok

    return true_positives/(true_positives+false_positives)


def calculate_AP(model_output, label):
    pass
