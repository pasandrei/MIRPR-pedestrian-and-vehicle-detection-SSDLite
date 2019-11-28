import torch
import datetime

from train.loss_fn import ssd_loss

def update_tensorboard_graphs(writer, loc_loss_train, class_loss_train, loc_loss_val, class_loss_val, epoch):
    writer.add_scalar('Localization Loss/train', loc_loss_train, epoch)
    writer.add_scalar('Classification Loss/train', class_loss_train, epoch)
    writer.add_scalar('Localization Loss/val', loc_loss_val, epoch)
    writer.add_scalar('Classification Loss/val', class_loss_val, epoch)

def evaluate(model, optimizer, anchors, grid_sizes, train_loader, valid_loader, losses, epoch, device, writer, params):
    '''
    evaluates model performance of the validation set, saves current set if it is better that the best so far
    '''
    loc_loss_train, class_loss_train = losses[2] / len(train_loader.dataset), losses[3] / len(train_loader.dataset)
    print('Average loss this epoch: Localization: {}; Classification: {}'.format(
        loc_loss_train, class_loss_train))
    print('Validation start...')

    loc_loss_val, class_loss_val = 0, 0

    BATCHES_TO_TEST = 1
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch_idx, (input_, label) in enumerate(valid_loader):
            print(datetime.datetime.now())
            input_ = input_.to(device)
            output = model(input_)
            l_loss, c_loss = ssd_loss(output, label, anchors, grid_sizes, device, params)
            loc_loss_val += l_loss.item()
            class_loss_val += c_loss.item()
            if batch_idx == BATCHES_TO_TEST:
                break

        # metric of performance... for now i take the loss
        SAVE_PATH = 'misc/experiments/{}/model_checkpoint'.format(params.model_id)
        loc_loss_val, class_loss_val = loc_loss_val / BATCHES_TO_TEST, class_loss_val / BATCHES_TO_TEST
        val_loss = loc_loss_val + class_loss_val
        if params.loss > val_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, SAVE_PATH)
            params.loss = val_loss
            params.save('misc/experiments/ssdnet/params.json')
            print('Model saved succesfully')


        # tensorboard
        update_tensorboard_graphs(writer, loc_loss_train, class_loss_train, loc_loss_val, class_loss_val, epoch)

    print('Validation finished')
