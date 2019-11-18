import torch
import torch.nn as nn
import torch.optim as optim

from train.config import Params
from data import dataloaders
from train import train
from architectures.models import SSDNet
from train.helpers import *

def test_anchor_mapping():
    '''
    assume for simplicity k=6 (# of anchors per feature map cell = FMC), batch = 1
    similarily, assume model only computes predictions from a 10x10 feature map

    things to consider:
    a) The network output format: 
    - the net computes a tensord of shape 10*10*6 x 4 for bounding box prediction
    so in this 600x4 for matrix, the first 6 lines correspond to the top left FMC, the next 6 to the one next to it and so on
    in the net we take a tensor of Bx(4*k)xHxW and spit out BxH*W*kx4
    so let's see if the ordering is preserved
    b) Correct anchor maps to FMC:
    - plot the anchor on the original image, as well as the FMC (just upsample)
    c) Plot the actual anchors
    '''

    def test_a():
        a = torch.rand(1,4*2,2,2)
        x = a.permute(0,2,3,1).contiguous()
        x = x.view(1,-1,4)

        print(a)
    
        print('---------------------------')

        print(x)

    test_a()
    def check_anchors():
        params = Params('misc/experiments/ssdnet/params.json')
        train_loader, valid_loader = dataloaders.get_dataloaders(params)
        anchors, grid_sizes = create_anchors()

        print(anchors, grid_sizes)
         
    #check_anchors()

test_anchor_mapping()