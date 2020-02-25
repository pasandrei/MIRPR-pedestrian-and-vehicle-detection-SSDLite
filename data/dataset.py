import torch
import os
import os.path
import torchvision.transforms.functional as F
import numpy as np
from data.vision_dataset import VisionDataset
from PIL import Image
from general_config.anchor_config import default_boxes
from utils.preprocessing import match, prepare_gt, get_bboxes

from albumentations import (
    RandomResizedCrop,
    HorizontalFlip,
    Blur,
    CLAHE,
    ChannelDropout,
    CoarseDropout,
    GaussNoise,
    RandomBrightnessContrast,
    RandomGamma,
    ToGray,
    Compose,
    BboxParams
)


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

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None, augmentation=True, params=None):
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.augmentation = augmentation
        self.params = params

        self.anchors_ltrb = default_boxes(order='ltrb')
        self.anchors_xywh = default_boxes(order='xywh')

        self.augmentations = self.get_aug([RandomResizedCrop(height=300, width=300, scale=(0.4, 1.0)),
                                           HorizontalFlip(), Blur(), CLAHE(), ChannelDropout(), CoarseDropout(),
                                           GaussNoise(), RandomBrightnessContrast(),
                                           RandomGamma(), ToGray(),
                                           ], min_visibility=0.3)

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
            orig_width, orig_height = img.size

            # get useful annotations
            bboxes, category_ids = get_bboxes(target)

            # augment data
            album_annotation = {'image': np.array(img), 'bboxes': bboxes, 'category_id': category_ids}
            augmented = self.augmentations(**album_annotation)
            image, bboxes, category_ids = augmented.values()

            # bring bboxes to correct format and check they are valid
            target = prepare_gt(image, bboxes, category_ids)
            self.check_bbox_validity(target)
            if target[0].nelement() == 0:
                continue

            # get image in right format - normalized tensor
            image = F.to_tensor(image)
            image = F.normalize(image, mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])

            # #anchors x 4 and #anchors x 1
            gt_bbox, gt_class = match(self.anchors_ltrb, self.anchors_xywh,
                                      target[0], target[1], self.params)

            imgs.append(image)
            targets_bboxes.append(gt_bbox)
            targets_classes.append(gt_class)
            image_info.append((img_id, (orig_width, orig_height)))

        # B x C x H x W
        batch_images = torch.stack(imgs)

        # B x #anchors x 4 and 1 respectively
        batch_bboxes = torch.stack(targets_bboxes)
        batch_class_ids = torch.stack(targets_classes)

        label = (batch_bboxes, batch_class_ids)

        return batch_images, label, image_info

    def __len__(self):
        return len(self.ids)

    def get_aug(self, aug, min_area=0., min_visibility=0.3):
        """
        Args:
        aug - set of albumentation augmentations
        min_area - minimum area to keep bbox
        min_visibility - minimum area percentage (to keep bbox) of original bbox after transform
        """
        return Compose(aug, bbox_params=BboxParams(format='coco', min_area=min_area,
                                                   min_visibility=min_visibility, label_fields=['category_id']))

    def check_bbox_validity(self, target):
        if target[0].nelement() == 0:
            return

        eps = 0.00001
        gt_bbox = target[0]

        # x and y must be positive
        col_1_ok = gt_bbox[:, 0] > 0
        col_2_ok = gt_bbox[:, 1] > 0

        # width and height must be strictly greater than zero
        col_3_ok = gt_bbox[:, 2] > eps
        col_4_ok = gt_bbox[:, 3] > eps

        # rows to keep
        ok = col_1_ok * col_2_ok * col_3_ok * col_4_ok
        target[0] = target[0][ok]
        target[1] = target[1][ok]
