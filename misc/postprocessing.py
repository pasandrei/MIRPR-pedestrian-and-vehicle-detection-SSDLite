import numpy as np
import cv2

# postprocess


def nms(boxes, overlap_threshold=0.6):
    """
    boxes: bounding boxes coordinates, ie, tuple of 4 integers
    overlap threshold: the threshold for which the overlapping images will be suppressed
    return the coordinates of the correct bounding boxes
    """

    # if there are no boxes, return an empty list
    if len(boxes) == 0:
        return []

    pick = []

    # grab the coordinates of the bounding boxes
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    # compute the area of the bounding boxes and sort the bounding
    # boxes by the bottom-right y-coordinate of the bounding box
    area = (x2 - x1 + 1) * (y2 - y1 + 1)
    idxs = np.argsort(y2)

    while len(idxs) > 0:
        # grab the last index in the indexes list, add the index
        # value to the list of picked indexes, then initialize
        # the suppression list (i.e. indexes that will be deleted)
        # using the last index
        last = len(idxs) - 1
        i = idxs[last]
        pick.append(i)
        suppress = [last]

        for pos in range(0, last):
            # grab the current index
            j = idxs[pos]

            # find the largest (x, y) coordinates for the start of
            # the bounding box and the smallest (x, y) coordinates
            # for the end of the bounding box
            xx1 = max(x1[i], x1[j])
            yy1 = max(y1[i], y1[j])
            xx2 = min(x2[i], x2[j])
            yy2 = min(y2[i], y2[j])

            # compute the width and height of the bounding box
            w = max(0, xx2 - xx1 + 1)
            h = max(0, yy2 - yy1 + 1)

            # compute the ratio of overlap between the computed
            # bounding box and the bounding box in the area list
            overlap = float(w * h) / area[j]

            # if there is sufficient overlap, suppress the
            # current bounding box
            if overlap > overlap_threshold:
                suppress.append(pos)

        # delete all indexes from the index list that are in the
        # suppression list
        idxs = np.delete(idxs, suppress)

    # return only the bounding boxes that were picked
    return boxes[pick]

#  postprocess


def plot_bounding_boxes(image, bounding_boxes, message='MUE', ok=0):
    """

    """
    # loop over the bounding boxes for each image and draw them
    image = image.transpose(1, 2, 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    color = (0, 255, 0) if ok else (0, 0, 255)
    for (startX, startY, endX, endY) in bounding_boxes:
        cv2.rectangle(image, (startY, startX), (endY, endX), color, 2)

    # display the image
    cv2.imshow(message, image)
    cv2.waitKey(0)
