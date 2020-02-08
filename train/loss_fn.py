import torch
import math
from torch import nn

from train.helpers import *
from general_config import classes_config

# inspired by fastai course


class BCE_Loss(nn.Module):
    def __init__(self, n_classes, device, focal_loss):
        super().__init__()
        self.n_classes = n_classes
        self.device = device
        self.id2idx = classes_config.training_ids2_idx
        self.focal_loss = focal_loss

    def forward(self, pred, targ):
        '''
        Arguments:
            pred - tensor of shape anchors x n_classes
            targ - tensor of shape anchors

        Explanation: computes weighted BCE loss between model prediction and target
            model predicts scores for each class, all 0s means background

        Returns: (weighted if focal) BCE loss
        '''
        t = []
        targ = targ.cpu().numpy()
        for clas_id in targ:
            bg = [0] * self.n_classes
            if clas_id != 100:
                bg[self.id2idx[clas_id]] = 1
            t.append(bg)

        t = torch.FloatTensor(t).to(self.device)
        weight = self.get_weight(pred, t) if self.focal_loss else None

        return torch.nn.functional.binary_cross_entropy_with_logits(pred, t, weight=weight, reduction='none')

    def get_weight(self, x, t):
        # focal loss decreases loss for correctly classified (P>0.5) examples, relative to the missclassified ones
        # thus increasing focus on them
        alpha, gamma = 0.25, 2.
        p = x.detach().sigmoid()

        # focal loss factor - decreases relative loss for well classified examples
        pt = p*t + (1-p)*(1-t)

        # counter positive/negative examples imbalance by assigning higher relative values to positives=1
        w = alpha*t + (1-alpha)*(1-t).to(self.device)

        # these two combined strongly encourage the network to predict a high value when
        # there is indeed a positive example
        return w * ((1-pt).pow(gamma))


class Detection_Loss():
    """
    Computes both localization and classification loss

    in args:
    anchors - #anchors x 4 cuda tensor
    grid_sizes - #anchors x 1 cuda tensor
    """

    def __init__(self, anchors, grid_sizes, device, params, focal_loss=False, hard_negative=False):
        self.anchors = anchors
        self.grid_sizes = grid_sizes
        self.device = device
        self.params = params
        self.hard_negative = hard_negative
        self.class_loss = BCE_Loss(params.n_classes, self.device, focal_loss)

        self.scale_xy = 10
        self.scale_wh = 5

    def ssd_1_loss(self, pred_bbox, pred_class, gt_bbox, gt_class):
        """
        Arguments:
            pred_bbox - #anchors x 4 cuda tensor - predicted bboxes for current image
            pred_class - #anchors x 2 cuda tensor - predicted class confidences for cur img
            gt_bbox - #obj x 4 cuda tensor - GT bboxes for objects in the cur img
            gt_class - #obj x 1 cuda tensor - class IDs for objects in cur img

        Explanation:
        model outputs offsets are converted to the final bbox predictions
        the matching phase is carried out
        localization (L1) and classification (BCE) loss are being computed and returned
        """
        # compute IOU for obj x anchor
        overlaps = jaccard(wh2corners(gt_bbox[:, :2], gt_bbox[:, 2:]), wh2corners(
            self.anchors[:, :2], self.anchors[:, 2:]))

        # map each anchor to the highest IOU obj, gt_idx - ids of mapped objects
        gt_bbox_for_matched_anchors, matched_gt_class_ids, pos_idx = map_to_ground_truth(
            overlaps, gt_bbox, gt_class, self.params)

        offsets = self.prepare_localization_offsets(gt_bbox_for_matched_anchors, pos_idx)

        loc_loss = self.localization_loss(pred_bbox, offsets, pos_idx)
        class_loss = self.classification_loss(pred_class, matched_gt_class_ids, pos_idx)

        return loc_loss, class_loss

    def ssd_loss(self, pred, targ):
        """
        Arguments:
            pred - model output - two tensors of dim B x #anchors x 4 and B x #anchors x n_classes in a list
            targ - ground truth - two tensors of dim B x #obj x 4 and B x #obj in a list

        Explanation:
        Loss will be calculated per image in the batch
        anchors will be mappend to overlapping GT bboxes higher than a threshold
        feature map cells corresponding to those anchors will have to predict those gt bboxes (loc loss)
        all feature map cells hape to predict a confidence (class loss)

        Return: loc and class loss per whole batch
        """

        localization_loss, classification_loss = 0., 0.

        for idx in range(pred[0].shape[0]):
            pred_bbox, pred_class = pred[0][idx], pred[1][idx]
            gt_bbox, gt_class = targ[0][idx].to(self.device), targ[1][idx].to(self.device)

            l_loss, c_loss = self.ssd_1_loss(pred_bbox, pred_class, gt_bbox, gt_class)
            localization_loss += l_loss
            classification_loss += c_loss

        return localization_loss, classification_loss

    def hard_negative_mining(self, losses, ids_for_anchors, ratio=3):
        """
        Taken from https://github.com/qfgaohao/pytorch-ssd
        """
        losses = losses.sum(dim=1)
        pos_mask = ids_for_anchors != 100
        num_pos = pos_mask.sum()
        num_neg = num_pos * ratio

        losses[pos_mask] = -math.inf
        _, indexes = losses.sort(descending=True)
        _, orders = indexes.sort()
        neg_mask = orders < num_neg
        return pos_mask | neg_mask

    def localization_loss(self, pred_bbox, offsets, pos_idx):
        """
        Arguments:
        pred_bbox - [#obj x 4] tensor - model predictions
        offsets - [#obj x 4] tensor - ground truth
        pos_idx - indeces of non background predicting anchors

        returns: l1 loss between predictions and ground truth divided by the number of matche anchors
        """
        matched_bbox = pred_bbox[pos_idx].float()
        return torch.nn.functional.smooth_l1_loss(matched_bbox, offsets,
                                                  reduction='sum') / pos_idx.shape[0]

    def classification_loss(self, pred_class, matched_gt_class_ids, pos_idx):
        """
        Arguments:
        pred_class - [#anchors x n_classes] tensor - confidence scores by each anchor
        matched_gt_class_ids - [#anchors x 1] tensor - ground truth class ids
        pos_idx - indeces of non background predicting anchors

        returns: binary cross entropy between predicted scores and one hot ground truth vectors,
        similarily normalized by the number of non background anchors
        """
        class_losses = self.class_loss(pred_class, matched_gt_class_ids)
        if self.hard_negative:
            loss = class_losses[self.hard_negative_mining(class_losses, matched_gt_class_ids)].sum()
        else:
            loss = class_losses.sum()
        return loss / pos_idx.shape[0]

    def prepare_localization_offsets(self, gt_bbox, pos_idx):
        """
        Arguments:
        - gt_bbox - [#matches_anchors x 4] tensor - matched ground truth bounding boxes
        - pos_idx - indeces of non background predicting anchors

        returns - offsets
        """
        matched_anchors = self.anchors[pos_idx]
        off_xy = self.scale_xy*(gt_bbox[:, :2] - matched_anchors[:, :2])/matched_anchors[:, 2:]
        off_wh = self.scale_wh*(gt_bbox[:, 2:]/matched_anchors[:, 2:]).log()
        return torch.cat((off_xy, off_wh), dim=1).contiguous()
