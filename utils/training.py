import torch
import numpy as np

from data import dataloaders

from architectures.models import SSDNet, resnet_ssd
from general_config import anchor_config
from train.optimizer_handler import *
from general_config import path_config


def update_tensorboard_graphs(writer, loc_loss_train, class_loss_train, loc_loss_val, class_loss_val, mAP, epoch):
    writer.add_scalar('Localization Loss/train', loc_loss_train, epoch)
    writer.add_scalar('Classification Loss/train', class_loss_train, epoch)
    writer.add_scalar('Localization Loss/val', loc_loss_val, epoch)
    writer.add_scalar('Classification Loss/val', class_loss_val, epoch)
    writer.add_scalar('mAP', mAP, epoch)


def gradient_weight_check(model):
    '''
    will pring mean abs value of gradients and weights during training to check for stability
    '''
    avg_grads, max_grads = [], []
    avg_weigths, max_weigths = [], []

    # try to understand comp graph better for why inter variables don't have grad retained and what this means for this stat
    for n, p in model.named_parameters():
        if (p.requires_grad) and not isinstance(p.grad, type(None)):
            avg_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
            avg_weigths.append(p.abs().mean())
            max_weigths.append(p.abs().max())

    avg_grads, max_grads = torch.FloatTensor(avg_grads), torch.FloatTensor(max_grads)
    avg_weigths, max_weigths = torch.FloatTensor(avg_weigths), torch.FloatTensor(max_weigths)

    return torch.mean(avg_grads), torch.mean(max_grads), torch.mean(avg_weigths), torch.mean(max_weigths)


def plot_grad_flow(model):
    # taken from Roshan Rane answer on pytorch forums
    '''Plots the gradients flowing through different layers in the net during training.
    Can be used for checking for possible gradient vanishing / exploding problems.

    Usage: Plug this function in Trainer class after loss.backwards() as
    "plot_grad_flow(self.model.named_parameters())" to visualize the gradient flow

    - will want to extend this to write ave_grads and max_grads to a simple csv file and plot progressions after training
    '''
    ave_grads = []
    max_grads = []
    layers = []
    for n, p in model.named_parameters():
        if(p.requires_grad) and ("bias" not in n):
            layers.append(n)
            ave_grads.append(p.grad.abs().mean())
            max_grads.append(p.grad.abs().max())
    plt.bar(np.arange(len(max_grads)), max_grads, alpha=0.1, lw=1, color="c")
    plt.bar(np.arange(len(max_grads)), ave_grads, alpha=0.1, lw=1, color="b")
    plt.hlines(0, 0, len(ave_grads)+1, lw=2, color="k")
    plt.xticks(range(0, len(ave_grads), 1), layers, rotation="vertical")
    plt.xlim(left=0, right=len(ave_grads))
    plt.ylim(bottom=-0.001, top=0.02)  # zoom in on the lower gradient regions
    plt.xlabel("Layers")
    plt.ylabel("average gradient")
    plt.title("Gradient flow")
    plt.grid(True)
    plt.legend([Line2D([0], [0], color="c", lw=4),
                Line2D([0], [0], color="b", lw=4),
                Line2D([0], [0], color="k", lw=4)], ['max-gradient', 'mean-gradient', 'zero-gradient'])


def model_setup(device, params):
    n_classes = params.n_classes if params.loss_type == "BCE" else params.n_classes + 1
    if params.model_id == 'ssdnet':
        model = SSDNet.SSD_Head(n_classes=n_classes, k_list=anchor_config.k_list)
    elif params.model_id == 'resnetssd':
        model = resnet_ssd.SSD300(n_classes=n_classes)
    model.to(device)

    return model


def optimizer_setup(model, device, params):
    if params.optimizer == 'adam':
        if params.freeze_backbone:
            optimizer = layer_specific_adam(model, params)
        else:
            optimizer = plain_adam(model, params)
    elif params.optimizer == 'sgd':
        if params.freeze_backbone:
            optimizer = layer_specific_sgd(model, params)
        else:
            optimizer = plain_sgd(model, params)

    return optimizer


def prepare_datasets(params):
    train_loader, valid_loader = dataloaders.get_dataloaders(params)
    return train_loader, valid_loader


def load_model(model, params, optimizer=None):
    checkpoint = torch.load('misc/experiments/{}/model_checkpoint'.format(params.model_id))
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint.get('epoch', 0)
    print('Model loaded successfully')

    return model, optimizer, start_epoch


def save_model(epoch, model, optimizer, params, stats, msg=None, by_loss=False):
    model_path = path_config.model_path
    if by_loss:
        model_path = path_config.model_path_loss
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, model_path.format(params.model_id))
    params.save(path_config.params_path.format(params.model_id))
    stats.save(path_config.stats_path.format(params.model_id))

    print(msg)
