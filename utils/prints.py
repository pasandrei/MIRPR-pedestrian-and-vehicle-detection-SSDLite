from general_config import anchor_config, general_config
from utils.training import gradient_weight_check
import datetime


def show_training_info(params):
    """
    prints training settings
    """
    params_ = params.dict
    for k, v in params_.items():
        print(str(k) + " : " + str(v))

    print("List of anchors per feature map cell: ", anchor_config.k_list)
    print("Model ID: ", general_config.model_id)
    print("-------------------------------------------------------")


def print_trained_parameters_count(model, optimizer):
    print('Total number of parameters of model: ', sum(p.numel() for p in model.parameters()))
    print('Total number of trainable parameters of model: ',
          sum(p.numel() for p in model.parameters() if p.requires_grad))

    print('Total number of parameters given to optimizer: ')
    print(sum(p.numel() for pg in optimizer.param_groups for p in pg['params']))
    print('Total number of trainable parameters given to optimizer: ')
    print(sum(p.numel() for pg in optimizer.param_groups for p in pg['params'] if p.requires_grad))
    print("-------------------------------------------------------")


def print_train_batch_stats(model, epoch, batch_idx, data_loader, losses, optimizer, params):
    '''
    prints statistics about the recently seen batches
    the printing interval is set through general_config.batch_stats_step - which means printing
    at an interval of the size of the dataloader divided by the steps

    eg: for a dataset of 1000 images, a batch size of 10 and batch_stats_step = 10
    - a print will be made after each 10 batches (100 images)
    '''
    one_nth_of_loader = len(data_loader) // general_config.batch_stats_step
    if (batch_idx + 1) % one_nth_of_loader == 0:
        print(datetime.datetime.now())
        print('Epoch: {} of {}'.format(epoch, params.n_epochs))
        print_batch_stats(batch_idx, data_loader, losses[0], losses[1], one_nth_of_loader, params)

        mean_grads, max_grads, mean_weights, max_weights = gradient_weight_check(model)
        print('Mean and max gradients over whole network: ', mean_grads, max_grads)
        print('Mean and max weights over whole network: ', mean_weights, max_weights)
        print("-------------------------------------------------------")

        last_lr = -1
        for pg in optimizer.param_groups:
            if pg['lr'] != last_lr:
                print('Current learning_rate:', pg['lr'])
            last_lr = pg['lr']
        losses[0], losses[1] = 0, 0


def print_val_batch_stats(model, batch_idx, data_loader, losses, params):
    one_nth_of_loader = len(data_loader) // general_config.batch_stats_step
    if (batch_idx + 1) % one_nth_of_loader == 0:
        print_batch_stats(batch_idx, data_loader, losses[0], losses[1], one_nth_of_loader, params)
        losses[0], losses[1] = 0, 0


def print_batch_stats(batch_idx, data_loader, loc_loss, class_loss, one_nth_of_loader, params):
    print('Batch: {} of {}'.format(batch_idx, len(data_loader)))

    avg_factor = one_nth_of_loader * params.batch_size
    print('Loss in the past {} samples: Localization {} Classification {}'.format(
        avg_factor, loc_loss / avg_factor, class_loss / avg_factor))


def print_dataset_stats(train_loader=None, valid_loader=None):
    if train_loader:
        print('Train size: ', len(train_loader), len(
            train_loader.dataset), len(train_loader.sampler.sampler))
    if valid_loader:
        print('Val size: ', len(valid_loader), len(
            valid_loader.dataset), len(valid_loader.sampler.sampler))

    print("-------------------------------------------------------")


def print_train_stats(train_loader, losses, params):
    """
    prints all epoch losses averaged on a single sample
    """
    eval_step_avg_factor = general_config.eval_step * len(train_loader.sampler.sampler)
    loc_loss_train, class_loss_train = losses[2] / \
        eval_step_avg_factor, losses[3] / eval_step_avg_factor

    print('Average train loss at eval start: Localization: {}; Classification: {}'.format(
        loc_loss_train, class_loss_train))
    return loc_loss_train, class_loss_train
