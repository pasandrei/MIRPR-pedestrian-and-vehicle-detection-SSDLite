import torch
from torch.utils.data import DataLoader
from data.dataset import CocoDetection
from general_config import constants, general_config


def collate_fn(batch):
    """Stack tensors into batches, keep image_info as a list of tuples."""
    images = torch.stack([item[0] for item in batch])
    bboxes = torch.stack([item[1] for item in batch])
    classes = torch.stack([item[2] for item in batch])
    infos = [item[3] for item in batch]
    return images, [bboxes, classes], infos


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

    return DataLoader(test_dataset, batch_size=params.batch_size,
                      shuffle=False, num_workers=general_config.num_workers,
                      persistent_workers=True, prefetch_factor=3,
                      pin_memory=True, collate_fn=collate_fn)


def get_dataloaders_test(params):
    return get_valid_dataloader(params)


def get_train_dataloader(params):
    train_annotations_path = constants.train_annotations_path
    train_dataset = CocoDetection(root=constants.train_images_folder,
                                  annFile=train_annotations_path,
                                  augmentation=True,
                                  params=params)

    return DataLoader(train_dataset, batch_size=params.batch_size,
                      shuffle=True, num_workers=general_config.num_workers,
                      persistent_workers=True, prefetch_factor=3,
                      pin_memory=True, drop_last=True, collate_fn=collate_fn)


def get_valid_dataloader(params):
    val_annotations_path = constants.val_annotations_path
    validation_dataset = CocoDetection(root=constants.val_images_folder,
                                       annFile=val_annotations_path,
                                       augmentation=False,
                                       params=params)

    return DataLoader(validation_dataset, batch_size=params.batch_size,
                      shuffle=False, num_workers=general_config.num_workers,
                      persistent_workers=True, prefetch_factor=3,
                      pin_memory=True, collate_fn=collate_fn)
