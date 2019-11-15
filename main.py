import torch
import torch.nn as nn
import torch.optim as optim

from train.config import Params
from data import dataloaders
from train import train
from architectures.models import SSDNet


def run(path='misc/experiments/ssdnet/params.json', resume=False):
    '''
    args: path - string path to the json config file
    trains model refered by that file, saves model and optimizer dict at the same location
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = Params('misc/experiments/ssdnet/params.json')

    if params.model_id == 'ssdnet':
        model = SSDNet.SSD_Head()
    model.to(device)

    if params.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate)

    if resume:
        # init model and optim dict
        pass

    train_loader, valid_loader = dataloaders.get_dataloaders()

    train.train(model, optimizer, train_loader, valid_loader, device, params)


run()
