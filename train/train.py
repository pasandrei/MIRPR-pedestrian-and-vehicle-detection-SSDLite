from train.loss_fn import ssd_loss
from train.helpers import *
from train.validate import evaluate
from train.lr_policies import constant_decay
from misc.print_stats import *
import datetime


def train_step(model, input_, label, anchors, grid_sizes, optimizer, losses, device, params):
    print(datetime.datetime.now())
    input_ = input_.to(device)

    optimizer.zero_grad()
    output = model(input_)
    l_loss, c_loss = ssd_loss(output, label, anchors, grid_sizes, device, params, input_)
    loss = l_loss + c_loss

    return

    update_losses(losses, l_loss.item(), c_loss.item())
    loss.backward()
    optimizer.step()


def train(model, optimizer, train_loader, valid_loader,
          device, params, start_epoch=0):
    '''
    args: model - nn.Module CNN to train
          optimizer - torch.optim
          params - json config
    trains model, saves best model by validation
    '''

    anchors, grid_sizes = create_anchors()
    anchors, grid_sizes = anchors.to(device), grid_sizes.to(device)

    print(datetime.datetime.now())
    for epoch in range(start_epoch, params.n_epochs):
        model.train()

        losses = [0] * 4
        for batch_idx, (input_, label) in enumerate(train_loader):
            train_step(model, input_, label, anchors, grid_sizes,
                       optimizer, losses, device, params)

            return


            '''
                Calculate AP for this batch
                add curent batch ap to counter
            '''

            if (batch_idx + 1) % params.train_stats_step == 0:
                print_batch_stats(model, epoch, batch_idx, train_loader,
                                  losses, params)
                losses[0], losses[1] = 0, 0
                ap_counter = 0

        if (epoch + 1) % params.eval_step == 0:
            evaluate(model, optimizer, anchors, grid_sizes, train_loader,
                     valid_loader, losses, epoch, device, params)
            losses[2], losses[3] = 0, 0

        # decay lr after epoch
        constant_decay.lr_decay(optimizer)

        for pg in optimizer.param_groups:
            print('Current learning_rate:', pg['lr'])


def update_losses(losses, l_loss, c_loss):
    '''
    losses[0], losses[1] - batch l and c loss, similarily for idx 2 and 3 epoch loss
    '''
    losses[0] += l_loss
    losses[1] += c_loss
    losses[2] += l_loss
    losses[3] += c_loss
