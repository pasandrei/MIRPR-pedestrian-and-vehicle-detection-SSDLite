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
    'fig_size': 320,
    'feat_size': [20, 10, 5, 3, 2, 1],
    'steps': [16, 32, 64, 107, 160, 320],
    'scales': [64, 112, 160, 208, 256, 304, 320],
    'aspect_ratios': [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]],
    'only_vertical': False,
    'reduce_first_layer': True
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

anchor_cfg = model_to_anchors[model_id]
fig_size = anchor_cfg['fig_size']
feat_size = anchor_cfg['feat_size']
steps = anchor_cfg['steps']
scales = anchor_cfg['scales']
aspect_ratios = anchor_cfg['aspect_ratios']
only_vertical = anchor_cfg['only_vertical']
reduce_first_layer = anchor_cfg.get('reduce_first_layer', False)

default_boxes = DefaultBoxes(fig_size, feat_size, steps,
                             scales, aspect_ratios, only_vertical=only_vertical,
                             reduce_first_layer=reduce_first_layer)

k_list = []
for i, aspect_ratio in enumerate(aspect_ratios):
    if only_vertical:
        k = len(aspect_ratio) + 2
    else:
        k = len(aspect_ratio) * 2 + 2
    if reduce_first_layer and i == 0:
        k -= 1  # drop the extra scale anchor
    k_list.append(k)

total_anchors = 0
for (size, k) in zip(feat_size, k_list):
    total_anchors += size*size*k
