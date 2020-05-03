from train.backbone_freezer import Backbone_Freezer
from utils.prints import print_train_batch_stats, print_train_stats
from general_config.general_config import device
from utils.training import update_losses, update_tensorboard_graphs, lr_decay_policy_setup
from general_config import general_config

import datetime

try:
    from apex import amp
except ImportError:
    raise ImportError("Please install APEX from https://github.com/nvidia/apex")


def train_step(model, input_, label, optimizer, losses, detection_loss, params, use_amp=False):
    input_ = input_.to(device)
    label[0] = label[0].to(device)
    label[1] = label[1].to(device)
    optimizer.zero_grad()
    output = model(input_)

    l_loss, c_loss = detection_loss.ssd_loss(output, label)
    loss = l_loss + c_loss

    update_losses(losses, l_loss.item(), c_loss.item())

    if use_amp:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    optimizer.step()


def train(model, optimizer, train_loader, model_evaluator,
          detection_loss, params, writer, start_epoch=0, use_amp=False):
    """
    args: model - nn.Module CNN to train
          optimizer - torch.optim
          train_loader - Dataloader object to provide data in batches
          model_evaluator - class used to validate model
          detection_loss - class used to handle loss
          params - json config
          writer - tensorboard writer - logs losses and mAP
    trains model, saves best model by validation
    """
    lr_decay_policy = lr_decay_policy_setup(params, len(train_loader))
    backbone_freezer = Backbone_Freezer(params)
    losses = [0] * 4

    if params.freeze_backbone:
        backbone_freezer.freeze_backbone(model)

    print(datetime.datetime.now())
    for epoch in range(start_epoch, params.n_epochs):
        model.train()

        backbone_freezer.step(epoch, model)
        print("Total number of parameters trained this epoch: ",
              sum(p.numel() for pg in optimizer.param_groups for p in pg['params'] if p.requires_grad))

        for batch_idx, (input_, label, _) in enumerate(train_loader):
            train_step(model, input_, label, optimizer, losses, detection_loss, params, use_amp)

            print_train_batch_stats(model=model, epoch=epoch, batch_idx=batch_idx,
                                    data_loader=train_loader,
                                    losses=losses, optimizer=optimizer, params=params)

            if epoch == 0 and params.warm_up:
                warm_up(train_loader, optimizer, params)
            else:
                lr_decay_policy.step(epoch, optimizer)

        if (epoch + 1) % general_config.eval_step == 0:
            mAP, loc_loss_val, class_loss_val = model_evaluator.complete_evaluate(model, optimizer,
                                                                                  epoch)
            loc_loss_train, class_loss_train = print_train_stats(
                train_loader, losses, params)
            update_tensorboard_graphs(writer, loc_loss_train, class_loss_train,
                                      loc_loss_val, class_loss_val, mAP, epoch)
            losses[2], losses[3] = 0, 0

        losses = [0] * 4


def warm_up(train_loader, optimizer, params):
    """
    linearly increase learning_rate 10x during the first epoch
    """
    train_size = len(train_loader)
    for pg in optimizer.param_groups:
        pg['lr'] += (1/train_size)*params.learning_rate*9
