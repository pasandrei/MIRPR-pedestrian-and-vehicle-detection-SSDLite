import torch
import math
from torch import nn
import copy

from general_config import classes_config
from utils.box_computations import *
from utils.training import *
from utils.preprocessing import *
from general_config.config import device
from general_config.anchor_config import default_boxes

# inspired by fastai course


class Classification_Loss(nn.Module):
    def __init__(self, params):
        super().__init__()
        self.id2idx = classes_config.training_ids2_idx
        self.loss_type = params.loss_type
        self.focal_loss = params.use_focal_loss

    def forward(self, pred, targ):
        '''
        Arguments:
            pred - tensor of shape batch x anchors x n_classes
            targ - tensor of shape anchors*batch

        Explanation: computes softmax loss between model prediction and target
            model predicts scores for each class, 0 is background class

        Returns: softmax loss or (weighted if focal) BCE loss
        '''
        n_classes = pred.shape[2]
        pred = pred.view(-1, n_classes)
        class_idx = self.map_id_to_idx(targ)

        if self.loss_type == "BCE":
            one_hot = torch.zeros((class_idx.shape[0], n_classes+1))
            one_hot = one_hot.to("cuda:0" if torch.cuda.is_available() else "cpu")
            one_hot[torch.arange(class_idx.shape[0]), class_idx] = 1

            # remove background column
            one_hot = one_hot[:, :-1]

            weight = self.get_weight(pred, one_hot) if self.focal_loss else None
            return torch.nn.functional.binary_cross_entropy_with_logits(pred, one_hot, weight=weight, reduction='none')
        else:
            return torch.nn.functional.cross_entropy(pred, class_idx, reduction='none')

    def get_weight(self, x, t):
        # focal loss decreases loss for correctly classified (P>0.5) examples, relative to the missclassified ones
        # thus increasing focus on them
        alpha, gamma = 0.25, 2.
        p = x.detach().sigmoid()

        # focal loss factor - decreases relative loss for well classified examples
        pt = p*t + (1-p)*(1-t)

        # counter positive/negative examples imbalance by assigning higher relative values to positives=1
        w = alpha*t + (1-alpha)*(1-t)

        # these two combined strongly encourage the network to predict a high value when
        # there is indeed a positive example
        return w * ((1-pt).pow(gamma))

    def map_id_to_idx(self, class_ids):
        """
        maps the tensor of class ids to indeces
        """
        class_idx = torch.zeros(class_ids.shape, dtype=int)
        for k, v in self.id2idx.items():
            class_idx[class_ids == k] = v

        class_idx = class_idx.to("cuda:0" if torch.cuda.is_available() else "cpu")
        return class_idx


class Detection_Loss():
    """
    Computes both localization and classification loss

    in args:
    anchors - #anchors x 4 cuda tensor
    grid_sizes - #anchors x 1 cuda tensor
    """

    def __init__(self, params):
        self.anchors_xywh = default_boxes(order="xywh")
        self.anchors_xywh = self.anchors_xywh.to(device)

        self.anchors_ltrb = default_boxes(order="ltrb")
        self.anchors_ltrb = self.anchors_ltrb.to(device)

        self.params = params
        self.hard_negative = params.use_hard_negative_mining
        self.class_loss = Classification_Loss(self.params)

        self.scale_xy = 10
        self.scale_wh = 5

    def match(self, gt_bbox, gt_class):
        """
        Arguments:
            gt_bbox - #obj x 4 tensor - GT bboxes for objects in the cur img
            gt_class - #obj x 1 tensor - class IDs for objects in cur img

        Explanation:
        argmax matching

        Returns:
        gt bboxes for each anchor (that mapped to an object)
        the tensor of class ids that each anchor has to predict
        indeces of object predicting anchors
        """
        # compute IOU for obj x anchor
        overlaps = jaccard(wh2corners(gt_bbox[:, :2], gt_bbox[:, 2:]), self.anchors_ltrb)

        # map each anchor to the highest IOU obj, gt_idx - ids of mapped objects
        gt_bbox_for_matched_anchors, matched_gt_class_ids, pos_idx = map_to_ground_truth(
            overlaps, gt_bbox, gt_class, self.params)

        return gt_bbox_for_matched_anchors, matched_gt_class_ids, pos_idx

    def ssd_loss(self, pred, targ):
        """
        Arguments:
            pred - model output - two tensors of dim B x #anchors x 4 and B x #anchors x n_classes in a list
            targ - ground truth - two tensors of dim B x #obj x 4 and B x #obj in a list

        Explanation:
        Matching is done per image
        loss is calculated per batch

        Return: loc and class loss per whole batch
        """
        batch_gt_bbox, batch_anchor_bbox, batch_pred_bbox, batch_class_ids = [], [], [], []

        for idx in range(pred[0].shape[0]):
            pred_bbox = pred[0][idx]

            gt_bbox, gt_class = targ[0][idx].to(device), targ[1][idx].to(device)
            gt_bbox_for_anchors, class_ids_for_anchors, pos_idx = self.match(gt_bbox, gt_class)

            batch_gt_bbox.append(gt_bbox_for_anchors)
            batch_anchor_bbox.append(self.anchors_xywh[pos_idx])
            batch_pred_bbox.append(pred_bbox[pos_idx])
            batch_class_ids.append(class_ids_for_anchors)

        # now we have everything in the batch
        batch_gt_bbox = torch.cat(batch_gt_bbox, dim=0)
        batch_anchor_bbox = torch.cat(batch_anchor_bbox, dim=0)
        batch_pred_bbox = torch.cat(batch_pred_bbox, dim=0)
        batch_class_ids = torch.cat(batch_class_ids, dim=0)

        # compute offsets
        offsets = self.prepare_localization_offsets(batch_gt_bbox, batch_anchor_bbox)

        # total matches
        norm_factor = batch_gt_bbox.shape[0]

        localization_loss = self.localization_loss(batch_pred_bbox, offsets, norm_factor)
        classification_loss = self.classification_loss(pred[1], batch_class_ids, norm_factor)

        return localization_loss, classification_loss

    def hard_negative_mining(self, losses, ids_for_anchors, ratio=3):
        """
        Taken from https://github.com/qfgaohao/pytorch-ssd
        """
        losses_ = copy.deepcopy(losses.detach())
        if self.params.loss_type == "BCE":
            losses_ = losses_.sum(dim=1)

        pos_mask = ids_for_anchors != 100
        num_pos = pos_mask.sum()
        num_neg = num_pos * ratio

        losses_[pos_mask] = -math.inf
        _, indexes = losses_.sort(descending=True)
        _, orders = indexes.sort()
        neg_mask = orders < num_neg
        return pos_mask | neg_mask

    def localization_loss(self, pred_bbox, offsets, norm_factor):
        """
        Arguments:
        pred_bbox - [#matches x 4] tensor - model predictions
        offsets - [#matches x 4] tensor - ground truth

        returns: l1 loss between predictions and ground truth divided by the number of matched anchors
        """
        return torch.nn.functional.smooth_l1_loss(pred_bbox, offsets,
                                                  reduction='sum') / norm_factor

    def classification_loss(self, pred_class, matched_gt_class_ids, norm_factor):
        """
        Arguments:
        pred_class - [#anchors x n_classes] tensor - confidence scores by each anchor
        matched_gt_class_ids - [#anchors x 1] tensor - ground truth class ids

        returns: softmax between predicted scores and one hot ground truth vectors,
        similarily normalized by the number of non background anchors
        """
        class_losses = self.class_loss(pred_class, matched_gt_class_ids)
        if self.hard_negative:
            loss = class_losses[self.hard_negative_mining(class_losses, matched_gt_class_ids)].sum()
        else:
            loss = class_losses.sum()

        return loss / norm_factor

    def prepare_localization_offsets(self, gt_bbox, matched_anchors):
        """
        Arguments:
        - gt_bbox - [#matches x 4] tensor - matched ground truth bounding boxes
        - matched_anchors - anchors from which these predictions are made

        returns - offsets
        """
        off_xy = self.scale_xy*(gt_bbox[:, :2] -
                                matched_anchors[:, :2])/matched_anchors[:, 2:]
        off_wh = self.scale_wh*(gt_bbox[:, 2:]/matched_anchors[:, 2:]).log()
        return torch.cat((off_xy, off_wh), dim=1).contiguous()
