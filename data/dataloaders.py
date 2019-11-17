import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from data.dataset import CocoDetection
from torch.utils.data.sampler import *

# get actual train and valid datasets, will be done when merging

def get_dataloaders(params):
    ''' creates and returns train and validation data loaders '''

    composed_transforms = transforms.Compose([transforms.ToTensor()])

    # train_dataset = torchvision.datasets.CocoDetection(root='../../COCO/train2017/train2017',
    #                                      annFile='../../COCO/annotations_trainval2017/annotations/instances_train2017.json',
    #                                       transform=composed_transforms)

    # train_dataloader = DataLoader(train_dataset, batch_size=1,
    #                        shuffle=True, num_workers=0)

    validation_dataset = CocoDetection(root='../../COCO/val2017/val2017',
                                       annFile='../../COCO/annotations_trainval2017/annotations/instances_val2017.json',
                                       transform=composed_transforms, transforms=None)


    valid_dataloader = DataLoader(validation_dataset, batch_size=None,
                                  shuffle=False, num_workers=0,
                                 sampler=BatchSampler(SequentialSampler([i for i in range(5000)]), batch_size=params.batch_size, drop_last=False))

    return valid_dataloader, valid_dataloader
