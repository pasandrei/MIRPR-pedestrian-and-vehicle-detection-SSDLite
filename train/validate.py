import torch
import datetime

from train.loss_fn import ssd_loss
from misc.metrics import calculate_AP


def update_tensorboard_graphs(writer, loc_loss_train, class_loss_train, loc_loss_val, class_loss_val, average_precision, epoch):
    writer.add_scalar('Localization Loss/train', loc_loss_train, epoch)
    writer.add_scalar('Classification Loss/train', class_loss_train, epoch)
    writer.add_scalar('Localization Loss/val', loc_loss_val, epoch)
    writer.add_scalar('Classification Loss/val', class_loss_val, epoch)
    writer.add_scalar('Precision', average_precision, epoch)


def evaluate(model, optimizer, anchors, grid_sizes, train_loader, valid_loader, losses, epoch, device, writer, params):
    '''
    evaluates model performance of the validation set, saves current set if it is better that the best so far
    '''
    loc_loss_train, class_loss_train = losses[2] / \
        len(train_loader.dataset), losses[3] / len(train_loader.dataset)
    print('Average loss this epoch: Localization: {}; Classification: {}'.format(
        losses[2] / len(train_loader.dataset), losses[3] / len(train_loader.dataset)))
    print('Validation start...')

    model.eval()
    with torch.no_grad():
        loc_loss_val, class_loss_val, sum_ap = 0, 0, 0

        for batch_idx, (input_, label) in enumerate(valid_loader):
            print(datetime.datetime.now())
            input_ = input_.to(device)
            output = model(input_)

            sum_ap += calculate_AP(output, label, anchors, grid_sizes)

            loc_loss, class_loss = ssd_loss(output, label, anchors, grid_sizes, device, params)
            loc_loss_val += loc_loss.item()
            class_loss_val += class_loss.item()

            if batch_idx % 50 == 0 and batch_idx > 0:
                print("Average precision: ", sum_ap / batch_idx, " until batch: ", batch_idx)


        SAVE_PATH = 'misc/experiments/{}/model_checkpoint'.format(params.model_id)
        average_precision = sum_ap / len(valid_loader.dataset)

        print("Validation average precision: ", average_precision)

        if params.average_precision < average_precision:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'average_precision': average_precision,
            }, SAVE_PATH)
            params.average_precision = average_precision
            params.save('misc/experiments/ssdnet/params.json')
            print('Model saved succesfully')

        # tensorboard
        update_tensorboard_graphs(writer, loc_loss_train, class_loss_train,
                                  loc_loss_val, class_loss_val, average_precision, epoch)

    print('Validation finished')
