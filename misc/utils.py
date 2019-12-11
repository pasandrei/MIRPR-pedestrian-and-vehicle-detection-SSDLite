import numpy as np
import cv2


def plot_bounding_boxes(image, bounding_boxes, message='no_message', size=(500, 500), ok=0):
    # loop over the bounding boxes for each image and draw them
    image = image.transpose(1, 2, 0)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, dsize=(size[1], size[0]))

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


def corners_to_wh(prediction_bboxes):
    for index in range(prediction_bboxes.shape[0]):
        prediction_bboxes[index][0], prediction_bboxes[index][1] = prediction_bboxes[index][1], prediction_bboxes[index][0]
        prediction_bboxes[index][2], prediction_bboxes[index][3] = prediction_bboxes[index][3], prediction_bboxes[index][2]

        width = prediction_bboxes[index][2] - prediction_bboxes[index][0]
        height = prediction_bboxes[index][3] - prediction_bboxes[index][1]

        prediction_bboxes[index][2] = width
        prediction_bboxes[index][3] = height

    return prediction_bboxes


def update_tensorboard_graphs(writer, loc_loss_train, class_loss_train, loc_loss_val, class_loss_val, average_precision, epoch):
    writer.add_scalar('Localization Loss/train', loc_loss_train, epoch)
    writer.add_scalar('Classification Loss/train', class_loss_train, epoch)
    writer.add_scalar('Localization Loss/val', loc_loss_val, epoch)
    writer.add_scalar('Classification Loss/val', class_loss_val, epoch)
    writer.add_scalar('Precision', average_precision, epoch)


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
                 "category_id": int(complete_outputs[index][4]), "id": prediction_id})

    return prediction_annotations, prediction_id


def evaluate_on_COCO_metrics(prediction_annotations):
    with open("fisierul.json", 'w') as f:
        json.dump(prediction_annotations, f)

    graundtrutu = COCO('..\\..\\COCO\\annotations\\instances_val2017.json')
    predictile = graundtrutu.loadRes(
        'C:\\Users\\Andrei Popovici\\Documents\\GitHub\\drl_zice_ca_se_poate_schimba_DA_MA\\fisierul.json')

    cocoevalu = COCOeval(graundtrutu, predictile, iouType='bbox')

    cocoevalu.evaluate()
    cocoevalu.accumulate()
    cocoevalu.summarize()

    return cocoevalu.summarize()._summarize(1)
