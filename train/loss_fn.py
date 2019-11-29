import torch
from torch import nn
import numpy as np
import random

from train.helpers import *
from misc.postprocessing import *


# inspired by fastai course
class BCE_Loss(nn.Module):
    def __init__(self, n_classes, device, norm_factor):
        super().__init__()
        self.n_classes = n_classes
        self.device = device
        self.id2idx = {1: 0, 3: 1}
        self.norm_factor = norm_factor

    def forward(self, pred, targ):
        '''
        pred - tensor of shape anchors x n_classes
        targ - tensor of shape anchors
        '''
        t = []
        targ = targ.cpu().numpy()
        for clas_id in targ:
            bg = [0] * self.n_classes
            if clas_id != 100:
                bg[self.id2idx[clas_id]] = 1
            t.append(bg)

        t = torch.FloatTensor(t).to(self.device)
        weight = self.get_weight(pred, t)
        return torch.nn.functional.binary_cross_entropy_with_logits(pred, t, weight=weight,
                                size_average=None, reduce=None, reduction='sum') / self.norm_factor

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


def ssd_1_loss(pred_bbox, pred_class, gt_bbox, gt_class, anchors, grid_sizes, device, params, image=None):
    # make network outputs same as gt bbox format
    pred_bbox = activations_to_bboxes(pred_bbox, anchors, grid_sizes)

    # compute IOU for obj x anchor
    overlaps = jaccard(gt_bbox, hw2corners(anchors[:, :2], anchors[:, 2:]))

    # map each anchor to the highest IOU obj, gt_idx - ids of mapped objects
    gt_bbox_for_matched_anchors, matched_gt_class_ids, matched_pred_bboxes, pos_idx = map_to_ground_truth(
        overlaps, gt_bbox, gt_class, pred_bbox)

    loc_loss = ((matched_pred_bboxes - gt_bbox_for_matched_anchors).abs()).mean()

    loss_f = BCE_Loss(params.n_classes, device, matched_pred_bboxes.shape[0])
    class_loss = loss_f(pred_class, matched_gt_class_ids)
    return loc_loss, class_loss


def ssd_loss(pred, targ, anchors, grid_sizes, device, params, image=None):
    '''
    args: pred - model output - two tensors of dim anchors x 4 and anchors x n_classes in a list
    targ - ground truth - two tensors of dim #obj x 4 and #obj in a list

    anchors will be mappend to overlapping GT bboxes, thus feature map cells corresponding to those anchors will have to predict those gt bboxes
    '''
    localization_loss, classification_loss = 0., 0.

    # computes the loss for each image in the batch
    for idx in range(pred[0].shape[0]):

        pred_bbox, pred_class = pred[0][idx], pred[1][idx]
        gt_bbox, gt_class = targ[0][idx].to(device), targ[1][idx].to(device)

        # assert that all tensors passed to ssd_1_loss are on GPU !!!!!!!!
        # assert pred_bbox.is_cuda is True
        # assert pred_class.is_cuda is True
        # assert gt_bbox.is_cuda is True
        # assert gt_class.is_cuda is True
        # assert anchors.is_cuda is True
        # assert grid_sizes.is_cuda is True

        l_loss, c_loss = ssd_1_loss(pred_bbox, pred_class, gt_bbox,
                                    gt_class, anchors, grid_sizes, device, params)
        localization_loss += l_loss
        classification_loss += c_loss

    return localization_loss, classification_loss
