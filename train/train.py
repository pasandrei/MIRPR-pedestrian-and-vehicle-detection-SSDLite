from train.loss_fn import ssd_loss
from train.helpers import *
from train.validate import evaluate
from train.lr_policies import constant_decay, retina_decay
from misc.print_stats import *

from misc.metrics import calculate_AP

import datetime


def train_step(model, input_, label, anchors, grid_sizes, optimizer, losses, device, params):
    # print(datetime.datetime.now())
    input_ = input_.to(device)

    optimizer.zero_grad()
    output = model(input_)
    l_loss, c_loss = ssd_loss(output, label, anchors, grid_sizes, device, params)
    loss = l_loss + c_loss

    update_losses(losses, l_loss.item(), c_loss.item())
    loss.backward()
    optimizer.step()

    # ap for batch
    return calculate_AP(output, label, anchors, grid_sizes)


def train(model, optimizer, train_loader, valid_loader,
          anchors, grid_sizes, writer, device, params, start_epoch=0):
    '''
    args: model - nn.Module CNN to train
          optimizer - torch.optim
          params - json config
    trains model, saves best model by validation
    '''
    lr_decay_policy = retina_decay.Lr_decay(params.learning_rate)
    losses = [0] * 4
    total_ap = 0
    one_tenth_of_loader = len(train_loader) // 10

    print(datetime.datetime.now())
    for epoch in range(start_epoch, params.n_epochs):
        model.train()

        for batch_idx, (input_, label) in enumerate(train_loader):
            cur_batch_ap = train_step(model, input_, label, anchors, grid_sizes,
                                      optimizer, losses, device, params)
            total_ap += cur_batch_ap

            if batch_idx % one_tenth_of_loader == 0 and batch_idx > 0:
                print_batch_stats(model, epoch, batch_idx, train_loader,
                                  losses, params)
                nr_images = (batch_idx + 1) * params.batch_size
                print("Average precision: ", total_ap / nr_images)
                losses[0], losses[1] = 0, 0
                for pg in optimizer.param_groups:
                    print('Current learning_rate:', pg['lr'])

        if (epoch + 1) % params.eval_step == 0:
            evaluate(model, optimizer, anchors, grid_sizes, train_loader,
                     valid_loader, losses, total_ap, epoch, device, writer, params)
            losses[2], losses[3], total_ap = 0, 0, 0

        # lr decay step
        lr_decay_policy.step(optimizer)

        #     SAVE_PATH = 'misc/experiments/{}/model_checkpoint'.format(params.model_id)
        #     eval_step_avg_factor = params.eval_step * len(train_loader.sampler.sampler)
        #     print("AVERAGES AT EVAL STEP: ",
        #           losses[2] / eval_step_avg_factor, losses[3] / eval_step_avg_factor)
        #     print('Average AP at eval step ', total_ap / eval_step_avg_factor)
        #     if params.loss > losses[2] + losses[3]:
        #         torch.save({
        #             'epoch': epoch,
        #             'model_state_dict': model.state_dict(),
        #             'optimizer_state_dict': optimizer.state_dict(),
        #             'loss': losses[2] + losses[3],
        #         }, SAVE_PATH)
        #         params.loss = losses[2] + losses[3]
        #         params.save('misc/experiments/ssdnet/params.json')
        #         print('Model saved succesfully')
        #     losses[2], losses[3], total_ap = 0, 0, 0
        #
        # constant_decay.lr_decay(optimizer)


def update_losses(losses, l_loss, c_loss):
    '''
    losses[0], losses[1] - batch l and c loss, similarily for idx 2 and 3 epoch loss
    '''
    losses[0] += l_loss
    losses[1] += c_loss
    losses[2] += l_loss
    losses[3] += c_loss
