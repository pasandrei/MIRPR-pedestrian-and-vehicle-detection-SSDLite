import os
import os.path
import torchvision.transforms.functional as F
import numpy as np
import random
from math import sqrt
from data.vision_dataset import VisionDataset
from PIL import Image
from general_config.anchor_config import default_boxes
from utils.preprocessing import match, prepare_gt, get_bboxes

from albumentations import (
    Resize,
    HorizontalFlip,
    RandomBrightnessContrast,
    Compose,
    BboxParams
)


def ssd_random_crop(image, bboxes, category_ids,
                    min_iou_choices=(0.0, 0.1, 0.3, 0.5, 0.7, 0.9, 1.0),
                    max_attempts=50):
    """
    TF Object Detection API style SSD random crop.

    Randomly selects a minimum IoU threshold, then samples crops until one
    satisfies the constraint against GT boxes. Keeps boxes whose center
    falls within the crop.

    Args:
        image: numpy array (H, W, 3)
        bboxes: list of [x, y, w, h] in COCO format (absolute pixel coords)
        category_ids: list of category IDs parallel to bboxes
    Returns:
        (cropped_image, filtered_bboxes, filtered_category_ids)
    """
    h, w = image.shape[:2]
    min_iou = random.choice(min_iou_choices)

    if min_iou == 1.0:
        return image, bboxes, category_ids

    # Convert GT boxes to xyxy for IoU computation
    gt_xyxy = []
    for b in bboxes:
        gt_xyxy.append([b[0], b[1], b[0] + b[2], b[1] + b[3]])

    for _ in range(max_attempts):
        # Sample crop dimensions
        ar = random.uniform(0.5, 2.0)
        area_frac = random.uniform(0.3, 1.0)
        crop_h = int(round(sqrt(h * w * area_frac / ar)))
        crop_w = int(round(crop_h * ar))
        if crop_w > w or crop_h > h or crop_w <= 0 or crop_h <= 0:
            continue

        # Sample crop position
        crop_x = random.randint(0, w - crop_w)
        crop_y = random.randint(0, h - crop_h)
        crop_xyxy = [crop_x, crop_y, crop_x + crop_w, crop_y + crop_h]

        # Check overlap constraint (fraction of GT box covered by crop)
        overlaps = []
        for gt in gt_xyxy:
            ix1 = max(crop_xyxy[0], gt[0])
            iy1 = max(crop_xyxy[1], gt[1])
            ix2 = min(crop_xyxy[2], gt[2])
            iy2 = min(crop_xyxy[3], gt[3])
            inter = max(0, ix2 - ix1) * max(0, iy2 - iy1)
            gt_area = (gt[2] - gt[0]) * (gt[3] - gt[1])
            overlaps.append(inter / gt_area if gt_area > 0 else 0)
        if max(overlaps) < min_iou:
            continue

        # Keep boxes whose center falls inside crop
        new_bboxes = []
        new_ids = []
        for b, cat_id, gt in zip(bboxes, category_ids, gt_xyxy):
            cx = b[0] + b[2] / 2
            cy = b[1] + b[3] / 2
            if crop_x <= cx <= crop_x + crop_w and crop_y <= cy <= crop_y + crop_h:
                # Clip to crop boundaries and adjust to crop origin
                nx1 = max(gt[0], crop_x) - crop_x
                ny1 = max(gt[1], crop_y) - crop_y
                nx2 = min(gt[2], crop_x + crop_w) - crop_x
                ny2 = min(gt[3], crop_y + crop_h) - crop_y
                nw = nx2 - nx1
                nh = ny2 - ny1
                if nw > 0 and nh > 0:
                    new_bboxes.append([nx1, ny1, nw, nh])
                    new_ids.append(cat_id)

        if len(new_bboxes) == 0:
            continue

        cropped = image[crop_y:crop_y + crop_h, crop_x:crop_x + crop_w]
        return cropped, new_bboxes, new_ids

    # All attempts failed, return original
    return image, bboxes, category_ids


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

    def _process_single(self, index):
        """Process a single image. Returns (image, gt_bbox, gt_class, info) or None if skipped."""
        coco = self.coco
        img_id = self.ids[index]
        ann_ids = coco.getAnnIds(imgIds=img_id)
        target = coco.loadAnns(ann_ids)
        path = coco.loadImgs(img_id)[0]['file_name']
        img = Image.open(os.path.join(self.root, path)).convert('RGB')
        orig_width, orig_height = img.size

        bboxes, category_ids = get_bboxes(target)
        bboxes, category_ids = self.check_bbox_validity(
            bboxes, category_ids, orig_width, orig_height)

        if self.run_type == "test":
            bboxes = [[3, 3, 100, 100]]
            category_ids = [0]
        if len(bboxes) == 0:
            return None

        if self.augmentation:
            image_np, bboxes, category_ids = ssd_random_crop(
                np.array(img), bboxes, category_ids)
            if len(bboxes) == 0:
                return None
            album_annotation = {'image': image_np, 'bboxes': bboxes,
                                'category_id': category_ids}
            transform_result = self.train_aug(**album_annotation)
        else:
            album_annotation = {'image': np.array(img), 'bboxes': bboxes,
                                'category_id': category_ids}
            transform_result = self.just_resize(**album_annotation)
        image, bboxes, category_ids = transform_result.values()

        if len(bboxes) == 0:
            return None

        target = prepare_gt(image, bboxes, category_ids)
        image = F.to_tensor(image)
        image = F.normalize(image, mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225])
        gt_bbox, gt_class = match(self.anchors_ltrb, self.anchors_xywh,
                                  target[0], target[1], self.params)

        return image, gt_bbox, gt_class, (img_id, (orig_width, orig_height))

    def __getitem__(self, index):
        """
        Returns a single (image, gt_bbox, gt_class, info) sample.
        If the requested index yields no valid bboxes, retries with
        random indices until a valid sample is found.
        """
        result = self._process_single(index)
        while result is None:
            index = random.randint(0, len(self.ids) - 1)
            result = self._process_single(index)
        return result

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
        train_aug = [
            Resize(height=self.params.input_height,
                   width=self.params.input_width),
            HorizontalFlip(),
            RandomBrightnessContrast(),
        ]
        self.train_aug = self.get_aug(train_aug, min_visibility=0.3)
        self.just_resize = self.get_aug([
            Resize(height=self.params.input_height,
                   width=self.params.input_width)])
