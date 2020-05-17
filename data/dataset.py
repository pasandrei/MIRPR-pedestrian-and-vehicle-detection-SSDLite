import torch
import os
import os.path
import torchvision.transforms.functional as F
import numpy as np
import random
from data.vision_dataset import VisionDataset
from PIL import Image
from general_config.anchor_config import default_boxes
from utils.preprocessing import match, prepare_gt, get_bboxes

from albumentations import (
    Resize,
    RandomResizedCrop,
    HorizontalFlip,
    Rotate,
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

    We are using the PyTorch COCO API on top of which we build our custom data processing
    """

    def __init__(self, root, annFile, transform=None,
                 target_transform=None, transforms=None, augmentation=True, params=None,
                 run_type="train"):
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))
        self.augmentation = augmentation
        self.params = params
        self.run_type = run_type

        self.init_augmentations()

        self.anchors_ltrb = default_boxes(order='ltrb')
        self.anchors_xywh = default_boxes(order='xywh')

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
            bboxes, category_ids = self.check_bbox_validity(
                bboxes, category_ids, orig_width, orig_height)

            if self.run_type == "test":
                # If we are using the official test dataset we must
                # not ignore images without annotations
                bboxes = [[3, 3, 100, 100]]
                category_ids = [0]
            if len(bboxes) == 0:
                continue

            album_annotation = {'image': np.array(
                img), 'bboxes': bboxes, 'category_id': category_ids}
            if self.augmentation:
                if random.random() > 0.5:
                    transform_result = self.crop_aug(**album_annotation)
                else:
                    transform_result = self.resize_aug(**album_annotation)
            else:
                transform_result = self.just_resize(**album_annotation)
            image, bboxes, category_ids = transform_result.values()

            # all bboxes might be lost after transform
            if len(bboxes) == 0:
                continue

            # bring bboxes to correct format
            target = prepare_gt(image, bboxes, category_ids)

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
                                                   min_visibility=min_visibility,
                                                   label_fields=['category_id']))

    def check_bbox_validity(self, bboxes, category_ids, width, height):
        """
        Some bboxes are invalid in COCO, have to filter them out otherwise albumentations will
        crash
        """
        eps = 0.000001
        valid_bboxes, valid_ids = [], []
        for bbox, id in zip(bboxes, category_ids):
            if bbox[0] <= eps or bbox[1] <= eps or (bbox[0] + bbox[2]) >= (width - eps) or (bbox[1] + bbox[3]) >= (height - eps):
                to_cut_x = max(0, -bbox[0])
                to_cut_y = max(0, -bbox[1])

                bbox[0] = max(0, bbox[0])
                bbox[1] = max(0, bbox[1])

                bbox[2] -= to_cut_x
                bbox[3] -= to_cut_y

                to_cut_x = min(0, width - (bbox[0] + bbox[2]))
                to_cut_y = min(0, height - (bbox[1] + bbox[3]))

                bbox[2] -= to_cut_x
                bbox[3] -= to_cut_y
            if bbox[2] * bbox[3] <= eps:
                continue

            valid_bboxes.append(bbox)
            valid_ids.append(id)

        return valid_bboxes, valid_ids

    def init_augmentations(self):
        common = [HorizontalFlip(), Rotate(limit=10),
                  RandomBrightnessContrast(),
                  ToGray(p=0.05)]

        random_crop_aug = [RandomResizedCrop(height=self.params.input_height,
                                             width=self.params.input_width,
                                             scale=(0.35, 1.0))]
        random_crop_aug.extend(common)

        simple_resize_aug = [Resize(height=self.params.input_height,
                                    width=self.params.input_width)]
        simple_resize_aug.extend(common)

        crop = self.get_aug(random_crop_aug, min_visibility=0.5)

        resize = self.get_aug(simple_resize_aug, min_visibility=0.5)

        just_resize = self.get_aug([Resize(height=self.params.input_height,
                                           width=self.params.input_width)])

        self.crop_aug = crop
        self.resize_aug = resize
        self.just_resize = just_resize
