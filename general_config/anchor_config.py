from general_config import general_config, constants
from utils.preprocessing import DefaultBoxes

model_id = general_config.model_id

# SSDLite modified
ssd_classic_19_19_vertical = {
    'fig_size': 300,
    'feat_size': [19, 10, 5, 3, 2, 1],
    'steps': [16, 32, 64, 100, 150, 300],
    'scales': [45, 99, 153, 207, 261, 280, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'only_vertical': True
}

# SSDLite
ssd_classic_19_19 = {
    'fig_size': 300,
    'feat_size': [19, 10, 5, 3, 2, 1],
    'steps': [16, 32, 64, 100, 150, 300],
    'scales': [45, 99, 153, 207, 261, 280, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'only_vertical': False
}

# classic SSD
ssd_classic = {
    'fig_size': 300,
    'feat_size': [38, 19, 10, 5, 3, 1],
    'steps': [8, 16, 32, 64, 100, 300],
    'scales': [21, 45, 99, 153, 207, 261, 315],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2], [2]],
    'only_vertical': False
}

model_to_anchors = {
    constants.ssd_modified: ssd_classic_19_19_vertical,
    constants.ssd: ssd_classic,
    constants.ssdlite: ssd_classic_19_19
}

fig_size, feat_size, steps, scales, aspect_ratios, only_vertical = model_to_anchors[model_id].values()

default_boxes = DefaultBoxes(fig_size, feat_size, steps,
                             scales, aspect_ratios, only_vertical=only_vertical)

k_list = [len(aspect_ratio)*2 + 2 for aspect_ratio in aspect_ratios]
if only_vertical:
    k_list = [len(aspect_ratio) + 2 for aspect_ratio in aspect_ratios]

total_anchors = 0
for (size, k) in zip(feat_size, k_list):
    total_anchors += size*size*k
