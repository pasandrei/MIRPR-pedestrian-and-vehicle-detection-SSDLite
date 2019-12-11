import torch
import datetime
import json

from pycocotools.cocoeval import COCOeval
from pycocotools.coco import COCO
from misc.model_output_handler import *
from misc.utils import *


def evaluate(model, optimizer, train_loader, valid_loader, losses, epoch, detection_loss, writer, params):
    '''
    evaluates model performance of the validation set, saves current set if it is better that the best so far
    '''
    output_handler = Model_output_handler(detection_loss.device)

    eval_step_avg_factor = params.eval_step * len(train_loader.sampler.sampler)
    loc_loss_train, class_loss_train = losses[2] / \
        eval_step_avg_factor, losses[3] / eval_step_avg_factor

    print('Average train loss at eval start: Localization: {}; Classification: {}'.format(
        loc_loss_train, class_loss_train))

    print('Validation start...')
    model.eval()
    with torch.no_grad():
        loc_loss_val, class_loss_val = 0, 0
        val_set_size = len(valid_loader.sampler.sampler)
        one_tenth_of_loader = len(valid_loader) // 1000

        prediction_annotations = []
        prediction_id = 0

        for batch_idx, (input_, label, image_info) in enumerate(valid_loader):
            # print(datetime.datetime.now())
            input_ = input_.to(detection_loss.device)
            output = model(input_)

            prediction_annotations, prediction_id = prepare_outputs_for_COCOeval(
                output, image_info, prediction_annotations, prediction_id, output_handler)

            loc_loss, class_loss = detection_loss.ssd_loss(output, label)
            loc_loss_val += loc_loss.item()
            class_loss_val += class_loss.item()

            if batch_idx % one_tenth_of_loader == 0 and batch_idx > 0:
                nr_images = (batch_idx + 1) * params.batch_size

                print("Average Loc Loss: ", loc_loss_val /
                      nr_images)
                print("Average Class Loss: ", class_loss_val /
                      nr_images, " until batch: ", batch_idx)

            print('should be good')

        SAVE_PATH = 'misc/experiments/{}/model_checkpoint'.format(params.model_id)

        mAP = evaluate_on_COCO_metrics(prediction_annotations)

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
    return mAP
