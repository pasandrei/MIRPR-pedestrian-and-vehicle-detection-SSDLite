from recordtype import recordtype
from general_config.general_config import device

import time


def train_step(model, input_, label, optimizer, losses, detection_loss, params, verbose, use_amp=False):
    input_ = input_.to(device)
    label[0] = label[0].to(device)
    label[1] = label[1].to(device)

    optimizer.zero_grad()

    # =================
    # START INFERENCE

    time_before_inference = time.time()

    output = model(input_)

    time_after_inference = time.time()
    inference_duration = time_after_inference - time_before_inference

    if verbose:
        print("Forward propagation time: {}".format(inference_duration))

    # END INFERENCE
    # =================

    # =================
    # START BACKPROPAGATION

    time_before_backprop = time.time()

    l_loss, c_loss = detection_loss.ssd_loss(output, label)
    loss = l_loss + c_loss

    update_losses(losses, l_loss.item(), c_loss.item())
    if use_amp:
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
    else:
        loss.backward()

    time_after_backprop = time.time()
    backprop_duration = time_after_backprop - time_before_backprop

    if verbose:
        print("Backward propagation time: {}".format(backprop_duration))

    # END BACKPROPAGATION
    # =================

    a = time.time()
    optimizer.step()
    b = time.time()
    optimizer_duration = b-a

    if verbose:
        print("Optimizer step time: ", optimizer_duration)

    Times = recordtype('Times', ['inference_time', 'backprop_time', 'optimizer_time'])
    batch_time = Times(inference_time=inference_duration,
                       backprop_time=backprop_duration, optimizer_time=optimizer_duration)
    return batch_time


def train(model, optimizer, train_loader, model_evaluator, detection_loss, params, verbose, start_epoch=0,
          use_amp=False):
    """
    args: model - nn.Module CNN to train
          optimizer - torch.optim
          params - json config
    trains model, saves best model by validation
    """

    losses = [0] * 4

    model.train()

    print("Total number of parameters trained this epoch: ",
          sum(p.numel() for pg in optimizer.param_groups for p in pg['params'] if p.requires_grad))

    average_time = recordtype('average_time', ['inference_time', 'backprop_time', 'optimizer_time'])
    total_time = recordtype('total_time', ['inference_time', 'backprop_time', 'optimizer_time'])

    average = average_time(inference_time=0, backprop_time=0, optimizer_time=0)
    total = total_time(inference_time=0, backprop_time=0, optimizer_time=0)

    WARM_UP = 2
    nr_batches = len(train_loader)
    counted_batches = nr_batches - WARM_UP
    for batch_idx, (input_, label, _) in enumerate(train_loader):
        if verbose:
            print("Batch id: ", batch_idx)
        now1 = time.time()
        batch_time = train_step(model, input_, label, optimizer,
                                losses, detection_loss, params, verbose, use_amp)
        now2 = time.time()

        if verbose:
            print("=================")

        if batch_idx >= WARM_UP:
            average.inference_time += batch_time.inference_time / counted_batches
            average.backprop_time += batch_time.backprop_time / counted_batches
            average.optimizer_time += batch_time.optimizer_time / counted_batches

            total.inference_time += batch_time.inference_time
            total.backprop_time += batch_time.backprop_time
            total.optimizer_time += batch_time.optimizer_time

        if verbose:
            print("Total time: {}\n\n".format(now2-now1))

    print("Times on {} batches of size {}:".format(counted_batches, params.batch_size))
    print(average)
    print(total)
    print("Total time: ", total.inference_time + total.backprop_time + total.optimizer_time)


def update_losses(losses, l_loss, c_loss):
    """
    losses[0], losses[1] - batch l and c loss, similarily for idx 2 and 3 epoch loss
    """
    losses[0] += l_loss
    losses[1] += c_loss
    losses[2] += l_loss
    losses[3] += c_loss
