import torch
from torch import nn

from train.loss_fn import ssd_loss


def evaluate(model, optimizer, valid_loader, epoch_loss, epoch, device, params):
    '''
    evaluates model performance of the validation set, saves current set if it is better that the best so far
    '''

    print('Average loss this epoch: ', epoch_loss / len(valid_loader))

    print('Validation start...')

    model.eval()
    with torch.no_grad():
        val_loss = 0.0
        for batch_idx, (input_, label) in enumerate(valid_loader):

            input_ = input_.to(device)
            label[0], label[1] = label[0].to(device), label[1].to(device)

            output = model(input_)
            loss = ssd_loss(output, label, device)
            val_loss += loss.item()

            # just testing
            if batch_idx == 10:
                break

        # metric of performance... for now i take the loss
        PATH = 'misc/experiments/{}/model_checkpoint'.format(params.model_id)
        if params.loss > val_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
            }, PATH)
            params.loss = val_loss
            params.save('misc/experiments/ssdnet/params.json')
            print('Model saved succesfully')
    print('Validation finished')
