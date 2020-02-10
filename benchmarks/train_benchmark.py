from train.helpers import *
from train.lr_policies import constant_decay, retina_decay
from train.backbone_freezer import Backbone_Freezer
from misc.print_stats import *

import datetime
import time


def train_step(model, input_, label, optimizer, losses, detection_loss, params):
    # print(datetime.datetime.now())
    input_ = input_.to(detection_loss.device)

    optimizer.zero_grad()
    now = time.time()
    output = model(input_)
    print(time.time() - now)
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

    losses = [0] * 4

    print(datetime.datetime.now())

    model.train()

    now = time.time()

    print("Total number of parameters trained this epoch: ",
          sum(p.numel() for pg in optimizer.param_groups for p in pg['params'] if p.requires_grad))

    for batch_idx, (input_, label, _) in enumerate(train_loader):
        now1 = time.time()
        print(batch_idx)
        train_step(model, input_, label, optimizer, losses, detection_loss, params)
        # if batch_idx == 3:
        now2 = time.time()

        print(now2-now1)

    later = time.time()
    print(later - now)
    # print((later - now2)*4/3)


def update_losses(losses, l_loss, c_loss):
    """
    losses[0], losses[1] - batch l and c loss, similarily for idx 2 and 3 epoch loss
    """
    losses[0] += l_loss
    losses[1] += c_loss
    losses[2] += l_loss
    losses[3] += c_loss
