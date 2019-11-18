import torch
import torch.nn as nn
import torch.optim as optim

from train.config import Params
from data import dataloaders
from train import train
from architectures.backbones import MobileNet
from train.helpers import visualize_data

import datetime

def measure_mobilenet():
    '''
    want to see how much time it takes for the mobileNet to process same chunk of data as SSDLite to see whether i messed up impl
    '''

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    params = Params('misc/experiments/ssdnet/params.json')

    model = MobileNet.mobilenet_v2()
    model.to(device)

    optimizer = optim.Adam(model.parameters())

    print('Total number of parameters of model: ', sum(p.numel() for p in model.parameters() if p.requires_grad))

    train_loader, valid_loader = dataloaders.get_dataloaders(params)
    
    dummy_targ = torch.rand(params.batch_size, 1)
    compress_all = nn.Conv2d(1280, 1, 1)
    compress_further = nn.Linear(100, 1)
    criterion = nn.BCELoss()
    sig = nn.Sigmoid()

    prev = datetime.datetime.now()
    print(prev)
    min_date_time = float('inf')

    for batch_idx, (input_, label) in enumerate(train_loader):
        input_ = input_.to(device)
            
        cur = datetime.datetime.now()
        print(cur)

        diff = cur - prev
        if divmod(diff.days * 86400 + diff.seconds, 60)[1] < min_date_time:
            min_date_time = divmod(diff.days * 86400 + diff.seconds, 60)[1]

        optimizer.zero_grad()
        output = model(input_)
        output[0].to(device)
        rez = output[1].to(device)
        rez = compress_all(rez)
        print(rez.shape)
        rez = rez.view(params.batch_size, 1, -1)
        rez = compress_further(rez)
        dummy_pred = sig(rez)
        loss = criterion(dummy_pred, dummy_targ)

        loss.backward()
        optimizer.step()

        if batch_idx == 10:
            break

    print(min_date_time)
