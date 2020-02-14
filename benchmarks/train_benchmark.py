from train.helpers import *
from train.lr_policies import constant_decay, retina_decay
from train.backbone_freezer import Backbone_Freezer
from misc.print_stats import *
from recordtype import recordtype

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

    print("Optimizer step time: ", optimizer_duration)

    Times = recordtype('Times', ['inference_time', 'backprop_time', 'optimizer_time'])
    batch_time = Times(inference_time=inference_duration,
                       backprop_time=backprop_duration, optimizer_time=optimizer_duration)
    return batch_time


def train(model, optimizer, train_loader, model_evaluator, detection_loss, params, start_epoch=0):
    """
    args: model - nn.Module CNN to train
          optimizer - torch.optim
          params - json config
    trains model, saves best model by validation
    """

    losses = [0] * 4

    a = time.time()
    b = time.time()
    print("{:.20f}".format(b-a))

    model.train()

    now = time.time()

    print("Total number of parameters trained this epoch: ",
          sum(p.numel() for pg in optimizer.param_groups for p in pg['params'] if p.requires_grad))

    Times = recordtype('average_time', ['inference_time', 'backprop_time', 'optimizer_time'])
    total = Times(inference_time=0, backprop_time=0, optimizer_time=0)

    WARM_UP = 2
    nr_batches = len(train_loader)
    counted_batches = nr_batches - WARM_UP
    for batch_idx, (input_, label, _) in enumerate(train_loader):
        print("Batch id: ", batch_idx)
        now1 = time.time()
        batch_time = train_step(model, input_, label, optimizer, losses, detection_loss, params)
        now2 = time.time()

        print("=================")
        if batch_idx >= WARM_UP:
            total.inference_time += batch_time.inference_time / counted_batches
            total.backprop_time += batch_time.backprop_time / counted_batches
            total.optimizer_time += batch_time.optimizer_time / counted_batches
        print("Total time: {}\n\n".format(now2-now1))

    print(total)


def update_losses(losses, l_loss, c_loss):
    """
    losses[0], losses[1] - batch l and c loss, similarily for idx 2 and 3 epoch loss
    """
    losses[0] += l_loss
    losses[1] += c_loss
    losses[2] += l_loss
    losses[3] += c_loss
