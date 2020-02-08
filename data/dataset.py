from data.vision_dataset import VisionDataset
from PIL import Image
import os
import os.path
import torchvision.transforms.functional as F
import numpy
import random
import copy

from train.helpers import *


class CocoDetection(VisionDataset):
    """`MS Coco Detection <http://mscoco.org/dataset/#detections-challenge2016>`_ Dataset.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.ToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.

    We are using the COCO API on top of which we build our custom data processing
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None, augmentation=True):
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.augmentation = augmentation

    def __getitem__(self, batched_indices):
        """
        return B x C x H x W image tensor and [B x img_bboxes, B x img_classes]
        """

        imgs, targets_bboxes, targets_classes, image_info = [], [], [], []
        for index in batched_indices:
            coco = self.coco
            img_id = self.ids[index]
            ann_ids = coco.getAnnIds(imgIds=img_id)
            target = coco.loadAnns(ann_ids)
            path = coco.loadImgs(img_id)[0]['file_name']
            img = Image.open(os.path.join(self.root, path)).convert('RGB')

            # target[0] = tensor of bboxes of objects in image
            # target[1] = tensor of class ids in image
            target = prepare_gt(img, target)
            self.check_bbox_validity(target)

            if target[0].nelement() == 0:
                continue

            width, height = img.size

            img = F.resize(img, size=(320, 320), interpolation=2)
            if self.augmentation:
                img, target = self.augment_data(img, target)

            # C x H x W
            img = F.to_tensor(img)
            img = F.normalize(img, mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

            imgs.append(img)
            targets_bboxes.append(target[0])
            targets_classes.append(target[1])
            image_info.append((img_id, (width, height)))

        # B x C x H x W
        batch_images = torch.stack(imgs)

        # batch_targets[0] = list of bboxes tensors for each image
        # batch_targets[1] = list of class id tensors for each image
        batch_targets = [targets_bboxes, targets_classes]

        return batch_images, batch_targets, image_info

    def __len__(self):
        return len(self.ids)

    def augment_data(self, img, target):
        # random flip
        if random.random() > 0.5:
            img = F.hflip(img)
            self.flip_gt_bboxes(target[0])

        # color jitter
        img = F.adjust_brightness(img, random.uniform(0.8, 1.2))
        img = F.adjust_contrast(img, random.uniform(0.8, 1.2))
        img = F.adjust_saturation(img, random.uniform(0.8, 1.2))
        img = F.adjust_hue(img, random.uniform(-0.05, 0.05))

        return img, target

    def flip_gt_bboxes(self, image_bboxes):
        # only mirror on x axis
        image_bboxes[:, 0] = 1 - image_bboxes[:, 0]

    def check_bbox_validity(self, target):
        eps = 0.00001
        gt_bbox = target[0]

        # x and y must be positive
        col_1_ok = gt_bbox[:, 0] > 0
        col_2_ok = gt_bbox[:, 1] > 0

        # width and height must be strictly greater than zero
        col_3_ok = gt_bbox[:, 0] > eps
        col_4_ok = gt_bbox[:, 1] > eps

        # rows to keep
        ok = col_1_ok * col_2_ok * col_3_ok * col_4_ok
        target[0] = target[0][ok]
        target[1] = target[1][ok]
