import torch
from torch import nn
# import torch.nn.functional as F

from train.helpers import *

# inspired by fastai course


class BCE_Loss(nn.Module):
    def __init__(self, num_classes, device):
        super().__init__()
        self.num_classes = num_classes
        self.device = device

    def forward(self, pred, targ):
        t = []
        for clas_id in targ:
            bg = [0, 0, 0]
            if clas_id == 3:
                clas_id = 2

            bg[clas_id] = 1
            t.append(bg)
        t = torch.FloatTensor(t).to(self.device)

        return torch.nn.functional.binary_cross_entropy_with_logits(pred, t, size_average=False)/self.num_classes

    def get_weight(self, x, t): return None


def ssd_1_loss(b_c, b_bb, bbox, clas, anchors, grid_sizes, device):
    # make network outputs same as gt bbox format
    a_ic = actn_to_bb(b_bb, anchors, grid_sizes)

    # get anchors in corner format too
    anchor_cnr = hw2corners(anchors[:, :2], anchors[:, 2:])

    # compute IOU for obj x anchor
    overlaps = jaccard(bbox.data, anchor_cnr.data)

    # map each anchor to the highest IOU obj, gt_idx - indexes of mapped objects
    gt_overlap, gt_idx = map_to_ground_truth(overlaps)

    # gt_clas[i] - ith anchor is matched with gt_clas[i] obj
    # the clas gt at this point is what class each anchor should predict
    gt_clas = clas[gt_idx]

    # anchors with a low IOU
    pos = gt_overlap > 0.4
    pos_idx = torch.nonzero(pos)[:, 0]

    # unmatched anchors should predict background
    gt_clas[~pos] = 2

    # get the bboxes of matched objects
    gt_bbox = bbox[gt_idx]

    #print('LOSS info')
    # print(pos_idx)
    # print(a_ic)
    # print(gt_bbox)
    # print(a_ic[pos_idx])
    # print(gt_bbox[pos_idx])
    loc_loss = ((a_ic[pos_idx] - gt_bbox[pos_idx]).abs()).mean()
    #print('LOSS info DONE')
    loss_f = BCE_Loss(3, device)
    clas_loss = loss_f(b_c, gt_clas)
    return loc_loss, clas_loss


def ssd_loss(pred, targ, device, batch_size=1):
    '''
    args: pred - model output - two tensors of dim anchors x 4 and anchors x n_classes in a list
    targ - ground truth - two tensors of dim #obj x 4 and #obj in a list
    '''
    lcs, lls = 0., 0.

    # create anchors
    anchors, grid_sizes = create_anchors()
    anchors, grid_sizes = anchors.to(device), grid_sizes.to(device)

    # computes the loss for each image in the batch
    for idx in range(batch_size):
        b_c, b_bb = pred[0][idx], pred[1][idx]
        bbox, clas = targ[0][idx], targ[1][idx]
        loc_loss, clas_loss = ssd_1_loss(b_c, b_bb, bbox, clas, anchors, grid_sizes, device)
        lls += loc_loss
        lcs += clas_loss

    # print("BUUUCII", lls.item(), lcs.item())
    return lls+lcs
