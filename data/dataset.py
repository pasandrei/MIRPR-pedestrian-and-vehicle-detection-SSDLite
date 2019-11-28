from data.vision_dataset import VisionDataset
from PIL import Image
import os
import os.path
import torchvision.transforms.functional as F
import numpy

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
    """

    def __init__(self, root, annFile, transform=None, target_transform=None, transforms=None):
        super().__init__(root, transforms, transform, target_transform)
        from pycocotools.coco import COCO
        self.coco = COCO(annFile)
        self.ids = list(sorted(self.coco.imgs.keys()))

    def __getitem__(self, batched_indices):
        """
        return B x C x H x W image tensor and [B x img_bboxes, B x img_classes]
        """

        imgs, targets_bboxes, targets_classes = [], [], []
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

            img = F.resize(img, size=(320, 320), interpolation=2)

            # C x H x W
            img = F.to_tensor(img)
            img = F.normalize(img, mean=[0.485, 0.456, 0.406],
                              std=[0.229, 0.224, 0.225])

            imgs.append(img)
            targets_bboxes.append(target[0])
            targets_classes.append(target[1])

        # B x C x H x W
        batch_images = torch.stack(imgs)

        # target[0] = list of bboxes tensors for each image
        # target[1] = list of class id tensors for each image
        batch_targets = [targets_bboxes, targets_classes]

        return batch_images, batch_targets

    def __len__(self):
        return len(self.ids)
