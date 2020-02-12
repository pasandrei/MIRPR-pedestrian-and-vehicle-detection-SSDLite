import json

from torch.utils.data import Dataset, DataLoader
from data.dataset import CocoDetection
from torch.utils.data.sampler import *
from general_config import path_config


def get_dataloaders(params):
    ''' creates and returns train and validation data loaders '''

    # train_dataloader = get_train_dataloader(batch_size=params.batch_size)
    valid_dataloader = get_valid_dataloader(batch_size=params.batch_size)

    return valid_dataloader, valid_dataloader


def get_dataloaders_test(params):
    return get_valid_dataloader(batch_size=params.batch_size)


def get_train_dataloader(batch_size):
    train_annotations_path = path_config.train_annotations_path
    train_dataset = CocoDetection(root=path_config.train_images_folder,
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
    val_annotations_path = path_config.val_annotations_path
    validation_dataset = CocoDetection(root=path_config.val_images_folder,
                                       annFile=val_annotations_path,
                                       augmentation=False)

    with open(val_annotations_path) as json_file:
        data = json.load(json_file)
        nr_images_in_val = len(data['images'])

    return DataLoader(validation_dataset, batch_size=None,
                      shuffle=False, num_workers=4,
                      sampler=BatchSampler(SequentialSampler([i for i in range(nr_images_in_val)]),
                                           batch_size=batch_size, drop_last=True))
