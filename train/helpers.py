import torch
import numpy as np
import cv2
from matplotlib import pyplot as plt


def hw2corners(ctr, hw):
    return torch.cat([ctr-hw/2, ctr+hw/2], dim=1)


def corners_to_center_hw(bbox):
    height = bbox[2]-bbox[0]
    width = bbox[3]-bbox[1]

    center_x = (bbox[2]+bbox[0])/2
    center_y = (bbox[3]+bbox[1])/2

    return torch.FloatTensor((center_x, center_y, height, width))


def clamp_corners(bbox):
    return torch.clamp(bbox, 0, 1)


def intersect(box_a, box_b):
    """ Returns the intersection of two boxes """
    max_xy = torch.min(box_a[:, None, 2:], box_b[None, :, 2:])
    min_xy = torch.max(box_a[:, None, :2], box_b[None, :, :2])
    inter = torch.clamp((max_xy - min_xy), min=0)
    return inter[:, :, 0] * inter[:, :, 1]


def box_sz(b):
    """ Returns the box size"""
    return ((b[:, 2]-b[:, 0]) * (b[:, 3]-b[:, 1]))


def jaccard(box_a, box_b):
    """ Returns the jaccard distance between two boxes"""
    inter = intersect(box_a, box_b)
    union = box_sz(box_a).unsqueeze(1) + box_sz(box_b).unsqueeze(0) - inter
    return inter / union


def activations_to_bboxes(actn, anchors, grid_sizes):
    """ activations to bounding boxes format """
    anchors = anchors.type(torch.float64)
    actn_offsets = torch.tanh(actn)

    actn_centers = actn_offsets[:, :2]/2 * grid_sizes + anchors[:, :2]
    actn_hw = (actn_offsets[:, 2:]/2+1) * anchors[:, 2:]

    return hw2corners(actn_centers, actn_hw)


def map_to_ground_truth(overlaps, gt_bbox, gt_class):
    """ maps priors to max IOU obj
   returns:
   - gt_bbox_for_matched_anchors: tensor of size matched_priors x 4 - essentially assigning GT bboxes to corresponding highest IOU priors
   - matched_gt_class_ids: tensor of size priors - where each value of the tensor indicates the class id that the priors feature map cell should predict
    """

    # for each object, what is the prior of maximum overlap
    gt_to_prior_overlap, gt_to_prior_idx = overlaps.max(1)

    # for each prior, what is the object of maximum overlap
    prior_to_gt_overlap, prior_to_gt_idx = overlaps.max(0)

    # for priors of max overlap, set a high value to make sure they match
    prior_to_gt_overlap[gt_to_prior_idx] = 1.99

    # for each prior, get the actual id of the class it should predict, unmatched anchors (low IOU) should predict background
    matched_gt_class_ids = gt_class[prior_to_gt_idx]
    pos = prior_to_gt_overlap > 0.5  # careful what you change!!!
    matched_gt_class_ids[~pos] = 100  # background code

    # for each matched prior, get the bbox it should predict
    raw_matched_bbox = gt_bbox[prior_to_gt_idx]
    pos_idx = torch.nonzero(pos)[:, 0]
    # which of those max values are actually precise enough?
    gt_bbox_for_matched_anchors = raw_matched_bbox[pos_idx]

    # so now we have the GT represented with priors
    return gt_bbox_for_matched_anchors, matched_gt_class_ids, pos_idx


def create_anchors():
    ''' anchors and sizes
    returns in the following format:
    k = zooms * ratios

    A = (grid_size1 ** 2 * k) + (grid_size2 ** 2 * k) +....+ (grid_sizen ** 2 * k) X 4

    where first k lines for this matrix are anchors centered in the top left corner cell of the first grid
    the next k lines are centered in the cell to the right of that (so they are generated by LINES)
    .
    .
    .
    after the first grid is finished comes the next and so on
    '''
    def create(anc_grids, anc_zooms, anc_ratios):
        anchor_scales = [(anz*i, anz*j) for anz in anc_zooms for (i, j) in anc_ratios]
        anc_offsets = [1/(o*2) for o in anc_grids]
        k = len(anchor_scales)

        anc_x = np.concatenate([np.repeat(np.linspace(ao, 1-ao, ag), ag)
                                for ao, ag in zip(anc_offsets, anc_grids)])
        anc_y = np.concatenate([np.tile(np.linspace(ao, 1-ao, ag), ag)
                                for ao, ag in zip(anc_offsets, anc_grids)])
        anc_ctrs = np.repeat(np.stack([anc_x, anc_y], axis=1), k, axis=0)

        anc_sizes = np.concatenate([np.array([[o/ag, p/ag] for i in range(ag*ag) for o, p in anchor_scales])
                                    for ag in anc_grids])

        grid_sizes = torch.from_numpy(np.concatenate([np.array([1/ag for i in range(ag*ag) for o, p in anchor_scales])
                                                      for ag in anc_grids])).unsqueeze(1)

        anchors = torch.from_numpy(np.concatenate([anc_ctrs, anc_sizes], axis=1)).float()

        return anchors, grid_sizes
        
    anc_grids10 = [20, 10, 5, 3, 2, 1]
    anc_zooms10 = [1., 1.25]
    anc_ratios10 = [(1., 1.), (2., 1.), (1., 2.)]

    anchors, grid_sizes = create(anc_grids10, anc_zooms10, anc_ratios10)

    anchor_corner = hw2corners(anchors[:, :2], anchors[:, 2:])
    anchor_corner = clamp_corners(anchor_corner)
    for idx, anchor in enumerate(anchor_corner):
        anchors[idx] = corners_to_center_hw(anchor)

    return anchors, grid_sizes


def prepare_gt(input_img, gt_target):
    '''
    NUMA SEFU STIE CE SENTAMPLA AICI. NICI MACAR DUMNEZEU NU ARE IDEEE
    args:
    - input_img: PIL image HxW
    - gt_target:

    bring gt bboxes in correct format and scales values to [0,1]

    return: gt[0] = tensor of bboxes of objects in image, gt[1] = tensor of class ids in image
    '''
    gt_bboxes, gt_classes = [], []
    for obj in gt_target:
        gt_bboxes.append(obj['bbox'])
        gt_classes.append(obj['category_id'])

    gt = [torch.FloatTensor(gt_bboxes), torch.IntTensor(gt_classes)]

    height_size, width_size = input_img.size[0], input_img.size[1]

    # width_size, height_size = 1, 1
    for idx, bbox in enumerate(gt[0]):
        new_bbox = [0] * 4
        new_bbox[1] = bbox[0] / height_size
        new_bbox[0] = bbox[1] / width_size
        new_bbox[3] = (bbox[0] + bbox[2]) / height_size
        new_bbox[2] = (bbox[1] + bbox[3]) / width_size
        gt[0][idx] = torch.FloatTensor(new_bbox)

    return gt
