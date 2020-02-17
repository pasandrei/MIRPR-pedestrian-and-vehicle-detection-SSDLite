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
            targ - tensor of shape batch x anchors

        Explanation: computes softmax loss between model prediction and target
            model predicts scores for each class, 0 is background class

        Returns: softmax loss or (weighted if focal) BCE loss
        '''
        bs, anchor_nr, n_classes = pred.shape
        pred = pred.view(-1, n_classes)
        targ = targ.view(-1)
        class_idx = self.map_id_to_idx(targ)

        if self.loss_type == "BCE":
            one_hot = torch.zeros((class_idx.shape[0], n_classes+1))
            one_hot = one_hot.to("cuda:0" if torch.cuda.is_available() else "cpu")
            one_hot[torch.arange(class_idx.shape[0]), class_idx] = 1

            # remove background column
            one_hot = one_hot[:, :-1]

            weight = self.get_weight(pred, one_hot) if self.focal_loss else None
            bce_loss = torch.nn.functional.binary_cross_entropy_with_logits(pred, one_hot,
                                                                            weight=weight,
                                                                            reduction='none')
            bce_loss = bce_loss.view(bs, anchor_nr, n_classes)
            return bce_loss.sum(dim=2)
        else:
            return torch.nn.functional.cross_entropy(pred, class_idx, reduction='none').view(bs, anchor_nr)

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

        self.anchors_batch = self.anchors.unsqueeze(dim=0).to(self.device)

    def ssd_loss(self, pred, targ):
        """
        Arguments:
            pred - model output - two tensors of dim B x #anchors x 4 and B x #anchors x n_classes in a list
            targ - ground truth - two tensors of dim B x #anchors x 4 and B x #anchors in a list

        Explanation:
        each image loss is normalized by the number of anchors to obj mappings
        total loss is normalized by the batch size

        Return: loc and class loss per whole batch
        """
        pred_bbox, pred_id = pred
        gt_bbox, gt_id = targ

        # compute offsets
        offsets = self.prepare_localization_offsets(gt_bbox)

        pos_mask = gt_id != 100
        pos_num = pos_mask.sum(dim=1)

        # B x 1
        localization_loss = self.localization_loss(pos_mask, pred_bbox, offsets)
        # B x 1
        classification_loss = self.classification_loss(pos_mask, pos_num, pred_id, gt_id)

        # normalize by mappings per each image in the batch then take the mean
        # we skip images without annotations, so no element in pos_num is 0
        localization_loss = (localization_loss / pos_num.float()).mean(dim=0)
        classification_loss = (classification_loss / pos_num.float()).mean(dim=0)
        return localization_loss, classification_loss

    def hard_negative_mining(self, pos_mask, pos_num, losses, ids_for_anchors, ratio=3):
        """
        Taken from https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD
        """
        losses_ = copy.deepcopy(losses.detach())
        losses_[pos_mask] = -math.inf
        _, indexes = losses_.sort(dim=1, descending=True)
        _, orders = indexes.sort(dim=1)
        neg_num = torch.clamp(ratio*pos_num, max=pos_mask.size(1)).unsqueeze(-1)
        neg_mask = orders < neg_num
        return pos_mask | neg_mask

    def localization_loss(self, pos_mask, pred_bbox, offsets):
        """
        Arguments:
        pos_mask - indeces of matched anchors
        pred_bbox - [B x #anchors x 4] tensor - model predictions
        offsets - [B x #anchors x 4] tensor - ground truth

        returns: l1 loss between predictions and ground truth batch_bboxes per each image in batch
        """
        # loss for each anchor
        loc_loss = (torch.nn.functional.smooth_l1_loss(
            pred_bbox, offsets, reduction='none')).sum(dim=2)

        # return the loss for those that actually matched
        return (loc_loss * pos_mask.float()).sum(dim=1)

    def classification_loss(self, pos_mask, pos_num, pred_id, gt_id):
        """
        Arguments:
        pos_mask - indeces of matched anchors
        pos_num - how many mappings for each image
        pred_id - [batch x #anchors x n_classes] tensor - confidence scores by each anchor
        gt_id - [batch x #anchors x 1] tensor - ground truth class ids

        returns: softmax/BCE between predicted scores and gt for each image in batch
        """

        class_losses = self.class_loss(pred_id, gt_id)
        if self.hard_negative:
            mask = self.hard_negative_mining(pos_mask, pos_num, class_losses, gt_id)
            loss = (class_losses * mask.float()).sum(dim=1)
        else:
            loss = class_losses.sum(dim=1)

        return loss

    def prepare_localization_offsets(self, gt_bbox):
        """
        Arguments:
        - gt_bbox - B x #anchors x 4 tensor - ground truth bounding boxes

        returns - offsets
        """
        off_xy = self.scale_xy*(gt_bbox[:, :, :2] -
                                self.anchors_batch[:, :, :2])/self.anchors_batch[:, :, 2:]
        off_wh = self.scale_wh*(gt_bbox[:, :, 2:]/self.anchors_batch[:, :, 2:]).log()
        return torch.cat((off_xy, off_wh), dim=2).contiguous()
