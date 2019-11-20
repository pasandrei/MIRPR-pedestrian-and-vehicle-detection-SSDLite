import torch
import torch.nn as nn
import torch.optim as optim

from train.config import Params
from data import dataloaders
from train import train
from architectures.models import SSDNet
from train.helpers import visualize_data


def run(path='misc/experiments/ssdnet/params.json', resume=False, visualize=False):
    '''
    args: path - string path to the json config file
    trains model refered by that file, saves model and optimizer dict at the same location
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = Params(path)

    if params.model_id == 'ssdnet':
        model = SSDNet.SSD_Head()
    model.to(device)

    for param in model.backbone.parameters():
        param.requires_grad = False

    if params.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate,
                               weight_decay=params.weight_decay)

    print('Number of epochs:', params.n_epochs)
    print('Total number of parameters of model: ', sum(p.numel()
                                                       for p in model.parameters() if p.requires_grad))
    #print('Total number of parameters given to optimizer: ', sum(p.numel() for p in optimizer.named_parameters()))

    start_epoch = 0
    if resume or visualize:
        checkpoint = torch.load('misc/experiments/{}/model_checkpoint'.format(params.model_id))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print('Model loaded successfully')

    train_loader, valid_loader = dataloaders.get_dataloaders(params)
    if visualize:
        visualize_data(valid_loader, model)
    else:
        train.train(model, optimizer, train_loader, valid_loader, device, params, start_epoch)


run()
