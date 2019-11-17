import torch
from torch import nn

from train.loss_fn import ssd_loss


def evaluate(model, optimizer, anchors, grid_sizes, train_loader, valid_loader, losses, epoch, device, params):
    '''
    evaluates model performance of the validation set, saves current set if it is better that the best so far
    '''

    print('Average loss this epoch: Localization: {}; Classification: {}'.format(losses[2] / len(train_loader), losses[3] / len(train_loader)))
    print('Validation start...')

    l_val_loss, c_val_loss = 0, 0
    model.eval()
    with torch.no_grad():
        val_loss = 0
        for batch_idx, (input_, label) in enumerate(valid_loader):
            input_ = input_.to(device)
            output = model(input_)
            l_loss, c_loss = ssd_loss(output, label, anchors, grid_sizes, device, params)
            l_val_loss += l_loss.item()
            c_val_loss += c_loss.item()

            if batch_idx == 10:
                break

        # metric of performance... for now i take the loss
        PATH = 'misc/experiments/{}/model_checkpoint'.format(params.model_id)
        val_loss = l_val_loss + c_val_loss
        if params.loss > val_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, PATH)
            params.loss = val_loss
            params.save('misc/experiments/ssdnet/params.json')
            print('Model saved succesfully')
    print('Validation finished')
