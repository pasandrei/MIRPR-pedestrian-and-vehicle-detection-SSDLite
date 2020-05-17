import json

from torch.utils.data import DataLoader
from data.dataset import CocoDetection
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler, SequentialSampler
from general_config import constants, general_config


def get_dataloaders(params):
    ''' creates and returns train and validation data loaders '''

    train_dataloader = get_train_dataloader(params)
    valid_dataloader = get_valid_dataloader(params)

    return train_dataloader, valid_dataloader


def get_test_dev(params):
    test_annotations_path = constants.test_annotations_path
    test_dataset = CocoDetection(root=constants.test_images_folder,
                                 annFile=test_annotations_path,
                                 augmentation=False,
                                 params=params,
                                 run_type="test")

    with open(test_annotations_path) as json_file:
        data = json.load(json_file)
        nr_images_in_test = len(data['images'])

    return DataLoader(test_dataset, batch_size=None,
                      shuffle=False, num_workers=general_config.num_workers,
                      sampler=BatchSampler(SubsetRandomSampler([i for i in range(nr_images_in_test)]),
                                           batch_size=params.batch_size, drop_last=False))


def get_dataloaders_test(params):
    return get_valid_dataloader(params)


def get_train_dataloader(params):
    train_annotations_path = constants.train_annotations_path
    train_dataset = CocoDetection(root=constants.train_images_folder,
                                  annFile=train_annotations_path,
                                  augmentation=True,
                                  params=params)

    with open(train_annotations_path) as json_file:
        data = json.load(json_file)
        nr_images_in_train = len(data['images'])

    return DataLoader(train_dataset, batch_size=None,
                      shuffle=False, num_workers=general_config.num_workers,
                      sampler=BatchSampler(SubsetRandomSampler([i for i in range(nr_images_in_train)]),
                                           batch_size=params.batch_size, drop_last=True))


def get_valid_dataloader(params):
    val_annotations_path = constants.val_annotations_path
    validation_dataset = CocoDetection(root=constants.val_images_folder,
                                       annFile=val_annotations_path,
                                       augmentation=False,
                                       params=params)

    with open(val_annotations_path) as json_file:
        data = json.load(json_file)
        nr_images_in_val = len(data['images'])

    return DataLoader(validation_dataset, batch_size=None,
                      shuffle=False, num_workers=general_config.num_workers,
                      sampler=BatchSampler(SequentialSampler([i for i in range(nr_images_in_val)]),
                                           batch_size=params.batch_size, drop_last=False))
