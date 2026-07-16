from train.backbone_freezer import Backbone_Freezer
from utils.prints import print_train_batch_stats, print_train_stats
from general_config.general_config import device
from utils.training import update_losses, update_tensorboard_graphs
from general_config import general_config, constants

import torch
import datetime


def train_step(model, input_, label, optimizer, losses, detection_loss, params, scaler=None,
               ema_model=None):
    input_ = input_.to(device, non_blocking=True)
    label[0] = label[0].to(device, non_blocking=True)
    label[1] = label[1].to(device, non_blocking=True)
    optimizer.zero_grad()

    if scaler is not None:
        with torch.amp.autocast('cuda'):
            output = model(input_)
            l_loss, c_loss = detection_loss.ssd_loss(output, label)
            loss = l_loss + c_loss
        update_losses(losses, l_loss.item(), c_loss.item())
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    else:
        output = model(input_)
        l_loss, c_loss = detection_loss.ssd_loss(output, label)
        loss = l_loss + c_loss
        update_losses(losses, l_loss.item(), c_loss.item())
        loss.backward()
        optimizer.step()

    if ema_model is not None:
        ema_model.update_parameters(model)


def train(model, optimizer, train_loader, model_evaluator,
          detection_loss, params, writer, lr_decay_policy, start_epoch=0, scaler=None,
          ema_model=None):
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

    backbone_freezer = Backbone_Freezer(params)
    losses = [0] * 4

    if params.freeze_backbone:
        backbone_freezer.freeze_backbone(model)

    print(datetime.datetime.now())
    for epoch in range(start_epoch, params.n_epochs):
        model.train()

        if general_config.model_id == constants.ssdlite:
            backbone_freezer.step(epoch, model)
        print("Total number of parameters trained this epoch: ",
              sum(p.numel() for pg in optimizer.param_groups for p in pg['params'] if p.requires_grad))

        for batch_idx, (input_, label, _) in enumerate(train_loader):
            warmup_epochs = params.warm_up if isinstance(params.warm_up, int) else (1 if params.warm_up else 0)
            if epoch < warmup_epochs:
                lr_decay_policy.warm_up(epoch, batch_idx, len(train_loader), warmup_epochs)
            else:
                lr_decay_policy.step(epoch)

            train_step(model, input_, label, optimizer, losses, detection_loss, params, scaler,
                       ema_model)

            print_train_batch_stats(model=model, epoch=epoch, batch_idx=batch_idx,
                                    data_loader=train_loader,
                                    losses=losses, optimizer=optimizer, params=params)

        if (epoch + 1) % general_config.eval_step == 0:
            mAP, loc_loss_val, class_loss_val = model_evaluator.complete_evaluate(model, optimizer,
                                                                                  epoch, ema_model)
            loc_loss_train, class_loss_train = print_train_stats(
                train_loader, losses, params)
            update_tensorboard_graphs(writer, loc_loss_train, class_loss_train,
                                      loc_loss_val, class_loss_val, mAP, epoch)
            losses[2], losses[3] = 0, 0

        losses[0], losses[1] = 0, 0
