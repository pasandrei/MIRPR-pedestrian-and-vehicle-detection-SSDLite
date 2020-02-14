from general_config import anchor_config
from utils.training import gradient_weight_check


def show_training_info(params):
    params_ = params.dict
    for k, v in params_.items():
        print(str(k) + " : " + str(v))

    print("List of anchors per feature map cell: ", anchor_config.k_list)
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


def print_batch_stats(model, epoch, batch_idx, train_loader, losses, params):
    '''
    prints statistics about the recently seen batches
    '''
    print('Epoch: {} of {}'.format(epoch, params.n_epochs))
    print('Batch: {} of {}'.format(batch_idx, len(train_loader)))

    # want to see per image stats
    one_tenth_of_loader = len(train_loader) // params.train_stats_step
    avg_factor = one_tenth_of_loader * params.batch_size
    print('Loss in the past {}th of the batches: Localization {} Classification {}'.format(
        params.train_stats_step, losses[0] / avg_factor, losses[1] / avg_factor))

    mean_grads, max_grads, mean_weights, max_weights = gradient_weight_check(model)

    print('Mean and max gradients over whole network: ', mean_grads, max_grads)
    print('Mean and max weights over whole network: ', mean_weights, max_weights)
    print("-------------------------------------------------------")


def print_dataset_stats(train_loader, valid_loader):
    print('Train size: ', len(train_loader), len(
        train_loader.dataset), len(train_loader.sampler.sampler))
    print('Val size: ', len(valid_loader), len(
        valid_loader.dataset), len(valid_loader.sampler.sampler))

    print("-------------------------------------------------------")
