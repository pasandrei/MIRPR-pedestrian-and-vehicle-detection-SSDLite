import torch
import datetime
import pycocotools.COCO

from train.loss_fn import ssd_loss
from misc.metrics import calculate_AP
from misc.postprocessing import convert_output_to_workable_data
from pycocotools.cocoeval import COCOeval


def update_tensorboard_graphs(writer, loc_loss_train, class_loss_train, loc_loss_val, class_loss_val, average_precision, epoch):
    writer.add_scalar('Localization Loss/train', loc_loss_train, epoch)
    writer.add_scalar('Classification Loss/train', class_loss_train, epoch)
    writer.add_scalar('Localization Loss/val', loc_loss_val, epoch)
    writer.add_scalar('Classification Loss/val', class_loss_val, epoch)
    writer.add_scalar('Precision', average_precision, epoch)


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

        for batch_idx, (input_, label) in enumerate(valid_loader):
            # print(datetime.datetime.now())
            input_ = input_.to(device)
            output = model(input_)

            prediction_bboxes, predicted_confidences = convert_output_to_workable_data(
                output, anchors, grid_sizes)

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

        val_loss = (loc_loss_train + loc_loss_val) / val_set_size
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
