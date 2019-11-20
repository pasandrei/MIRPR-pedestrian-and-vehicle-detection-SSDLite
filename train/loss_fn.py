import torch
from torch import nn
# import torch.nn.functional as F

from train.helpers import *


# inspired by fastai course
class BCE_Loss(nn.Module):
    def __init__(self, n_classes, device):
        super().__init__()
        self.n_classes = n_classes
        self.device = device

    def forward(self, pred, targ):
        '''
        pred - tensor of shape anchors x n_classes
        targ - tensor of shape anchors
        '''
        t = []
        for clas_id in targ:
            bg = torch.zeros(3).to(self.device)
            if clas_id.item() == 1:
                bg[0] = 1
            elif clas_id.item() == 3:
                bg[1] = 1
            else:
                bg[2] = 1
            t.append(bg)
        weight = self.get_weight(pred, t)
        return torch.nn.functional.binary_cross_entropy_with_logits(pred, t, weight)

    def get_weight(self, x, t):
        alpha, gamma = 0.9, 2.
        p = x.detach()
        # confidence of prediction
        pt = p*t + (1-p)*(1-t)

        # non-background / background weight
        w = torch.FloatTensor([1, 1, 1]).to(self.device)

        # complete weighing factor
        return w * ((1-pt).pow(gamma))


def ssd_1_loss(pred_bbox, pred_class, gt_bbox, gt_class, anchors, grid_sizes, device):
    # make network outputs same as gt bbox format
    pred_bbox = activations_to_bboxes(pred_bbox, anchors, grid_sizes)

    # compute IOU for obj x anchor
    overlaps = jaccard(gt_bbox, hw2corners(anchors[:, :2], anchors[:, :2]))

    # map each anchor to the highest IOU obj, gt_idx - ids of mapped objects
    matched_gt_bbox, matched_gt_class_ids, pos_idx=map_to_ground_truth(
        overlaps, gt_bbox, gt_class)

    loc_loss=((pred_bbox[pos_idx] - matched_gt_bbox).abs()).mean()

    loss_f=BCE_Loss(3, device)
    class_loss=loss_f(pred_class, matched_gt_class_ids)
    return loc_loss, class_loss


def ssd_loss(pred, targ, anchors, grid_sizes, device, params):
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
        assert pred_bbox.is_cuda() == True
        assert pred_class.is_cuda() == True
        assert gt_bbox.is_cuda() == True
        assert gt_class.is_cuda() == True
        assert anchors.is_cuda() == True
        assert grid_sizes.is_cuda() == True

        l_loss, c_loss=ssd_1_loss(pred_bbox, pred_class, gt_bbox,
                                    gt_class, anchors, grid_sizes, device)
        localization_loss += l_loss
        classification_loss += c_loss

    return localization_loss, classification_loss
