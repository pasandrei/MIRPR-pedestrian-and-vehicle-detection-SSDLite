import torch
import torch.optim as optim

from train.config import Params
from train.validate import evaluate
from train.helpers import *
from data import dataloaders
from train import train
from architectures.models import SSDNet
from train.loss_fn import Detection_Loss


from torch.utils.tensorboard import SummaryWriter


def run(path='misc/experiments/ssdnet/params.json', resume=False, eval_only=False):
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

    if params.optimizer == 'adam':
        optimizer = optim.Adam(model.parameters(), lr=params.learning_rate,
                               weight_decay=params.weight_decay)
    elif params.optimizer == 'sgd':
        optimizer = optim.SGD(model.parameters(), lr=params.learning_rate,
                              weight_decay=params.weight_decay, momentum=0.9)

    print('Number of epochs:', params.n_epochs)
    print('Total number of parameters of model: ',
          sum(p.numel() for p in model.parameters() if p.requires_grad))
    print('Total number of parameters given to optimizer: ')

    opt_params = 0
    for pg in optimizer.param_groups:
        opt_params += sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(opt_params)

    start_epoch = 0
    if resume or eval_only:
        checkpoint = torch.load('misc/experiments/{}/model_checkpoint'.format(params.model_id))
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch']
        print('Model loaded successfully')

        for pg in optimizer.param_groups:
            pg['lr'] = 0.05

    train_loader, valid_loader = dataloaders.get_dataloaders(params)

    print('Train size: ', len(train_loader), len(
        train_loader.dataset), len(train_loader.sampler.sampler))
    print('Val size: ', len(valid_loader), len(
        valid_loader.dataset), len(valid_loader.sampler.sampler))

    writer = SummaryWriter(filename_suffix=params.model_id)
    anchors, grid_sizes = create_anchors()
    anchors, grid_sizes = anchors.to(device), grid_sizes.to(device)

    detection_loss = Detection_Loss(anchors, grid_sizes, device, params)

    if eval_only:
        print('Only eval')
        for name, param in model.named_parameters():
            if param.requires_grad:
                print(name, param.data)
        losses = [0, 0, 0, 0]
        epoch, total_ap = 0, 0
        evaluate(model, optimizer, train_loader,
                 valid_loader, losses, total_ap, epoch, detection_loss, writer, params)
    else:
        train.train(model, optimizer, train_loader, valid_loader,
                    writer, detection_loss, params, start_epoch)
# run()
