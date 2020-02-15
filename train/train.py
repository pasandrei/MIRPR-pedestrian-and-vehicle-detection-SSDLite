from train.lr_policies import retina_decay
from train.backbone_freezer import Backbone_Freezer
from utils.training import *
from utils.prints import print_batch_stats
from general_config.config import device

import datetime


def train_step(model, input_, label, optimizer, losses, detection_loss, params):
    input_ = input_.to(device)

    optimizer.zero_grad()
    output = model(input_)
    l_loss, c_loss = detection_loss.ssd_loss(output, label)
    loss = l_loss + c_loss

    update_losses(losses, l_loss.item(), c_loss.item())
    loss.backward()
    optimizer.step()


def train(model, optimizer, train_loader, model_evaluator, detection_loss, params, start_epoch=0):
    """
    args: model - nn.Module CNN to train
          optimizer - torch.optim
          params - json config
    trains model, saves best model by validation
    """
    lr_decay_policy = retina_decay.Lr_decay(
        lr=params.learning_rate, start_epoch=start_epoch, params=params)
    backbone_freezer = Backbone_Freezer(params)
    losses = [0] * 4
    one_tenth_of_loader = len(train_loader) // params.train_stats_step

    if params.freeze_backbone:
        backbone_freezer.freeze_backbone(model)

    print(datetime.datetime.now())
    for epoch in range(start_epoch, params.n_epochs):
        model.train()

        backbone_freezer.step(epoch, model)
        lr_decay_policy.step(optimizer)
        print("Total number of parameters trained this epoch: ",
              sum(p.numel() for pg in optimizer.param_groups for p in pg['params'] if p.requires_grad))

        for batch_idx, (input_, label, _) in enumerate(train_loader):
            train_step(model, input_, label, optimizer, losses, detection_loss, params)

            if (batch_idx + 1) % one_tenth_of_loader == 0:
                print(datetime.datetime.now())
                print_batch_stats(model, epoch, batch_idx, train_loader,
                                  losses, params)
                losses[0], losses[1] = 0, 0
                for pg in optimizer.param_groups:
                    print('Current learning_rate:', pg['lr'])

            if epoch == 0 and params.warm_up:
                warm_up(train_loader, optimizer, params)

        if (epoch + 1) % params.eval_step == 0:
            model_evaluator.complete_evaluate(model, optimizer, train_loader, losses, epoch)

        losses = [0] * 4


def update_losses(losses, l_loss, c_loss):
    """
    losses[0], losses[1] - batch l and c loss, similarily for idx 2 and 3 epoch loss
    """
    losses[0] += l_loss
    losses[1] += c_loss
    losses[2] += l_loss
    losses[3] += c_loss


def warm_up(train_loader, optimizer, params):
    """
    linearly increase learning_rate 10x during the first epoch
    """
    train_size = len(train_loader)
    for pg in optimizer.param_groups:
        pg['lr'] += (1/train_size)*params.learning_rate*9
