from utils.preprocessing import DefaultBoxes

fig_size = 300
feat_size = [19, 10, 5, 3, 2, 1]
steps = [16, 32, 64, 100, 150, 300]
# use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
scales = [45, 99, 153, 207, 261, 290, 315]
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]

k_list = [len(aspect_ratio)*2 + 2 for aspect_ratio in aspect_ratios]

total_anchors = 0
for (size, k) in zip(feat_size, k_list):
    total_anchors += size*size*k

default_boxes = DefaultBoxes(fig_size, feat_size, steps, scales, aspect_ratios)
