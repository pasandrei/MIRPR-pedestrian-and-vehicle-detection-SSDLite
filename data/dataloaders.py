import torchvision
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from data.dataset import CocoDetection
from torch.utils.data.sampler import *
import json


def get_dataloaders(params):
    ''' creates and returns train and validation data loaders '''

    train_dataloader = get_train_dataloader(params)
    valid_dataloader = get_valid_dataloader(params)

    return train_dataloader, valid_dataloader


def get_dataloaders_test(params):
    return get_valid_dataloader


def get_train_dataloader(batch_size):
    train_annotations_path = 'C:\\Users\Andrei Popovici\Desktop\COCO\\annotations\\instances_train2017.json'
    train_dataset = CocoDetection(root='C:\\Users\Andrei Popovici\Desktop\COCO\\train2017',
                                  annFile=train_annotations_path,
                                  augmentation=True)

    with open(train_annotations_path) as json_file:
        data = json.load(json_file)
        nr_images_in_train = len(data['images'])

    return DataLoader(train_dataset, batch_size=None,
                      shuffle=False, num_workers=4,
                      sampler=BatchSampler(SubsetRandomSampler([i for i in range(nr_images_in_train)]),
                                           batch_size=batch_size, drop_last=True))


def get_valid_dataloader(batch_size):
    val_annotations_path = 'C:\\Users\Andrei Popovici\Desktop\COCO\\annotations\\instances_val2017.json'
    validation_dataset = CocoDetection(root='C:\\Users\Andrei Popovici\Desktop\COCO\\val2017',
                                       annFile=val_annotations_path,
                                       augmentation=False)

    with open(val_annotations_path) as json_file:
        data = json.load(json_file)
        nr_images_in_val = len(data['images'])

    return DataLoader(validation_dataset, batch_size=None,
                      shuffle=False, num_workers=4,
                      sampler=BatchSampler(SequentialSampler([i for i in range(nr_images_in_val)]),
                                           batch_size=batch_size, drop_last=True))
