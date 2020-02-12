from train.helpers import *
from train.lr_policies import constant_decay, retina_decay
from train.backbone_freezer import Backbone_Freezer
from misc.print_stats import *
from collections import namedtuple

import datetime
import time


def train_step(model, input_, label, optimizer, losses, detection_loss, params):
    input_ = input_.to(detection_loss.device)

    optimizer.zero_grad()

    # =================
    # START INFERENCE

    time_before_inference = time.time()

    output = model(input_)

    time_after_inference = time.time()
    inference_duration = time_after_inference - time_before_inference
    print("Forward propagation time: {}".format(inference_duration))

    # END INFERENCE
    # =================

    # =================
    # START BACKPROPAGATION

    time_before_backprop = time.time()

    l_loss, c_loss = detection_loss.ssd_loss(output, label)
    loss = l_loss + c_loss

    update_losses(losses, l_loss.item(), c_loss.item())
    loss.backward()

    time_after_backprop = time.time()
    backprop_duration = time_after_backprop - time_before_backprop

    print("Backward propagation time: {}".format(backprop_duration))

    # END BACKPROPAGATION
    # =================

    a = time.time()
    optimizer.step()
    b = time.time()
    optimizer_duration = b-a

    print("Optimizer step time: ", b-a)

    Times = namedtuple('Times', ['inference_time', 'backprop_time', 'optimizer_time'])
    t = Times(inference_time=inference_duration, backprop_time=backprop_duration, optimizer_time=optimizer_duration)
    return t


def train(model, optimizer, train_loader, model_evaluator, detection_loss, params, start_epoch=0):
    """
    args: model - nn.Module CNN to train
          optimizer - torch.optim
          params - json config
    trains model, saves best model by validation
    """

    losses = [0] * 4

    a= time.time()
    b=time.time()
    print("{:.20f}".format(b-a))

    model.train()

    now = time.time()

    print("Total number of parameters trained this epoch: ",
          sum(p.numel() for pg in optimizer.param_groups for p in pg['params'] if p.requires_grad))

    for batch_idx, (input_, label, _) in enumerate(train_loader):
        print("Batch id: ", batch_idx)
        now1 = time.time()
        t = train_step(model, input_, label, optimizer, losses, detection_loss, params)
        now2 = time.time()

        print("=================")
        print(t)
        print("Total time: {}\n\n".format(now2-now1))

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
