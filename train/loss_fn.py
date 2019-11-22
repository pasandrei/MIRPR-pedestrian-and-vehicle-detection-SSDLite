import torch
from torch import nn
import numpy as np
# import torch.nn.functional as F

from train.helpers import *
# system.append()
from misc.postprocessing import *


# inspired by fastai course
class BCE_Loss(nn.Module):
    def __init__(self, n_classes, device):
        super().__init__()
        self.n_classes = n_classes
        self.device = device
        self.id2idx = {1: 0, 3: 1, 100: 2}

    def forward(self, pred, targ):
        '''
        pred - tensor of shape anchors x n_classes
        targ - tensor of shape anchors
        '''
        t = []
        targ = targ.cpu().numpy()
        for clas_id in targ:
            bg = [0] * self.n_classes
            bg[self.id2idx[clas_id]] = 1
            t.append(bg)

        t = torch.FloatTensor(t).to(self.device)
        weight = self.get_weight(pred, t)
        return torch.nn.functional.binary_cross_entropy_with_logits(pred, t, weight)

    def get_weight(self, x, t):
        alpha, gamma = 0.9, 3.
        p = x.detach()
        # confidence of prediction
        pt = p*t + (1-p)*(1-t)

        # non-background / background weight
        w = torch.FloatTensor([1, 1, 0.01]).to(self.device)

        # complete weighing factor
        return w * ((1-pt).pow(gamma))


def ssd_1_loss(pred_bbox, pred_class, gt_bbox, gt_class, anchors, grid_sizes, device, image=None):
    # make network outputs same as gt bbox format
    pred_bbox = activations_to_bboxes(pred_bbox, anchors, grid_sizes)
    print("CACAT")
    print(jaccard(torch.FloatTensor([[0, 0, 50, 50], [0, 0, 25, 25]])/100,
                  torch.FloatTensor([[25, 25, 50, 50], [25, 25, 100, 100]])/100))

    # compute IOU for obj x anchor
    overlaps = jaccard(gt_bbox, hw2corners(anchors[:, :2], anchors[:, 2:]))
    print("OVERLAPS", overlaps)
    anchors = hw2corners(anchors[:, :2], anchors[:, 2:])

    # map each anchor to the highest IOU obj, gt_idx - ids of mapped objects
    matched_gt_bbox, matched_gt_class_ids, matched_pred_bbox, pos_idx = map_to_ground_truth(
        overlaps, gt_bbox, gt_class, pred_bbox)

    loc_loss = ((matched_pred_bbox - matched_gt_bbox).abs()).mean()
    print(matched_gt_bbox.shape)

    a = (matched_gt_bbox.detach().cpu().numpy()*320).astype(int)
    print("matched_gt_bbox")
    print(a)

    image = (image * 255).cpu().numpy().astype(np.uint8)

    pula = (anchors.cpu().numpy() * 320).astype(int)

    curent_pulici = []
    for index, pulica in enumerate(pula):
        if index < 3000:
            continue

        if index % 6 == 0:
            print("\n**************\n", overlaps[0][index])
            curent_pulici.append(pulica)

        if index >= 3110:
            break

    curent_pulici = np.array(curent_pulici)

    pzd = pula[:10]
    print("PZD: ", pzd)
    pula = pula[pos_idx.cpu().numpy()]
    print("pos_idx")
    print(pos_idx)
    plot_bounding_boxes(image, curent_pulici)
    plot_bounding_boxes(image, a)

    loss_f = BCE_Loss(3, device)
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
        cur_img = image[idx]

        pred_bbox, pred_class = pred[0][idx], pred[1][idx]
        gt_bbox, gt_class = targ[0][idx].to(device), targ[1][idx].to(device)

        # assert that all tensors passed to ssd_1_loss are on GPU !!!!!!!!
        assert pred_bbox.is_cuda is True
        assert pred_class.is_cuda is True
        assert gt_bbox.is_cuda is True
        assert gt_class.is_cuda is True
        assert anchors.is_cuda is True
        assert grid_sizes.is_cuda is True

        l_loss, c_loss = ssd_1_loss(pred_bbox, pred_class, gt_bbox,
                                    gt_class, anchors, grid_sizes, device, cur_img)
        localization_loss += l_loss
        classification_loss += c_loss

    return localization_loss, classification_loss
