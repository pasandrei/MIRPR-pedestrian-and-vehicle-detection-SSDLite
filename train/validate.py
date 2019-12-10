import torch
import datetime
import json

from train.loss_fn import ssd_loss
from misc.postprocessing import convert_output_to_workable_data, after_nms, predictions_over_threshold, get_predicted_class, corners_to_wh
from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO


def update_tensorboard_graphs(writer, loc_loss_train, class_loss_train, loc_loss_val, class_loss_val, average_precision, epoch):
    writer.add_scalar('Localization Loss/train', loc_loss_train, epoch)
    writer.add_scalar('Classification Loss/train', class_loss_train, epoch)
    writer.add_scalar('Localization Loss/val', loc_loss_val, epoch)
    writer.add_scalar('Classification Loss/val', class_loss_val, epoch)
    writer.add_scalar('Precision', average_precision, epoch)


def prepare_outputs_for_COCOeval(output, anchors, grid_sizes, image_info, prediction_annotations, prediction_id):
    batch_size = output[0].shape[0]

    for i in range(batch_size):
        prediction_bboxes, predicted_confidences = convert_output_to_workable_data(
            output[0][i], output[1][i], anchors, grid_sizes, image_info[i][1])

        prediction_bboxes, predicted_confidences = predictions_over_threshold(
            prediction_bboxes, predicted_confidences, 0.25)

        prediction_bboxes, predicted_confidences = after_nms(
            prediction_bboxes, predicted_confidences)

        prediction_bboxes = corners_to_wh(prediction_bboxes)
        prediction_index, predicted_confidences = get_predicted_class(predicted_confidences)

        image_id = image_info[i][0]

        # print("SHAPE de prediction idnex:", prediction_index.shape)

        for index in range(prediction_bboxes.shape[0]):
            if prediction_index[index] == 0:
                category_id = 1
            else:
                category_id = 3

            python_category_id = [int(x) for x in prediction_bboxes[index]]

            prediction_id += 1
            prediction_annotations.append(
                {"image_id": image_id, "bbox": python_category_id,
                 "score": float(predicted_confidences[index]),
                 "category_id": category_id, "id": prediction_id})

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


def evaluate(model, optimizer, anchors, grid_sizes, train_loader, valid_loader, losses, total_ap, epoch, device, writer, params):
    '''
    evaluates model performance of the validation set, saves current set if it is better that the best so far
    '''
    eval_step_avg_factor = params.eval_step * len(train_loader.sampler.sampler)
    loc_loss_train, class_loss_train = losses[2] / \
        eval_step_avg_factor, losses[3] / eval_step_avg_factor

    print('Average train loss at eval start: Localization: {}; Classification: {}'.format(
        loc_loss_train, class_loss_train))

    ap = total_ap / eval_step_avg_factor
    print('Average train precision at eval start: {}'.format(ap))

    print('Validation start...')

    model.eval()
    with torch.no_grad():
        loc_loss_val, class_loss_val = 0, 0
        val_set_size = len(valid_loader.sampler.sampler)
        one_tenth_of_loader = len(valid_loader) // 10

        prediction_annotations = []
        prediction_id = 0

        for batch_idx, (input_, label, image_info) in enumerate(valid_loader):
            # print(datetime.datetime.now())
            input_ = input_.to(device)
            output = model(input_)

            prediction_annotations, prediction_id = prepare_outputs_for_COCOeval(
                output, anchors, grid_sizes, image_info, prediction_annotations, prediction_id)

            loc_loss, class_loss = ssd_loss(output, label, anchors, grid_sizes, device, params)
            loc_loss_val += loc_loss.item()
            class_loss_val += class_loss.item()

            if batch_idx % one_tenth_of_loader == 0 and batch_idx > 0:
                nr_images = (batch_idx + 1) * params.batch_size

                print("Average Loc Loss: ", loc_loss_val /
                      nr_images)
                print("Average Class Loss: ", class_loss_val /
                      nr_images, " until batch: ", batch_idx)

        SAVE_PATH = 'misc/experiments/{}/model_checkpoint'.format(params.model_id)

        evaluate_on_COCO_metrics(prediction_annotations)

        val_loss = (class_loss_val + loc_loss_val) / val_set_size
        if params.loss > val_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }, SAVE_PATH)
            params.loss = val_loss
            params.save('misc/experiments/ssdnet/params.json')
            print('Model saved succesfully')

        # tensorboard
        average_precision = 0
        update_tensorboard_graphs(writer, loc_loss_train, class_loss_train,
                                  loc_loss_val, class_loss_val, average_precision, epoch)

    print('Validation finished')
