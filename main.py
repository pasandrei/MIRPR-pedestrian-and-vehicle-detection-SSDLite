import torch
import torch.optim as optim

from train.config import Params
from data import dataloaders
from train import train
from architectures.models import SSDNet
from configparser import ConfigParser

config = ConfigParser()
config.read("config.ini")

def run(path=config["PARAMS"]["ssdnet"], resume=False, visualize=False):
    '''
    args: path - string path to the json config file
    trains model refered by that file, saves model and optimizer dict at the same location
    '''
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    params = Params(path)
    print("MODEL ID: ", params.model_id)
    if params.model_id == 'ssdnet' or params.model_id == 'ssdnet_loc':
        model = SSDNet.SSD_Head(n_classes=params.n_classes)
    model.to(device)

    # for param_group in model.parameters():
    #     param_group.requires_grad = False

    if params.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate,
                               weight_decay=params.weight_decay)
    print('Number of epochs:', params.n_epochs)
    print('Total number of parameters of model: ',
          sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('Total number of parameters given to optimizer: ')

    opt_params = 0
    for pg in optimizer.param_groups:
        opt_params += sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(opt_params)

    start_epoch = 0
    if resume:
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


# run()
