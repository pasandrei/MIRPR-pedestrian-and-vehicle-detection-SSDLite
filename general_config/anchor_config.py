from utils.preprocessing import DefaultBoxes

# SSDLite small anchors config
# print("SMALLL ANCHRS")
# fig_size = 300
# feat_size = [19, 10, 5, 3, 2, 1]
# steps = [16, 32, 64, 100, 150, 300]
# # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
# scales = [20, 75, 110, 150, 190, 250, 300]
# aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]

# Different set of small
# fig_size = 300
# feat_size = [19, 10, 5, 3, 2, 1]
# steps = [16, 32, 64, 100, 150, 300]
# # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
# scales = [25, 50, 120, 175, 230, 270, 300]
# aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]


# SSDLite medium anchors config
# print("MEDIUM ANCHRS")
# fig_size = 300
# feat_size = [19, 10, 5, 3, 2, 1]
# steps = [16, 32, 64, 100, 150, 300]
# # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
# scales = [30, 85, 125, 170, 210, 260, 300]
# aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]

# SSDLite big anchors config
print("BIG ANCHRS")
fig_size = 300
feat_size = [19, 10, 5, 3, 2, 1]
steps = [16, 32, 64, 100, 150, 300]
# use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
scales = [45, 99, 153, 207, 261, 280, 315]
aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2, 3], [2, 3]]

# ORIGINAL SSD Anchor config
# fig_size = 300
# feat_size = [38, 19, 10, 5, 3, 1]
# steps = [8, 16, 32, 64, 100, 300]
# # use the scales here: https://github.com/amdegroot/ssd.pytorch/blob/master/data/config.py
# scales = [21, 45, 99, 153, 207, 261, 315]
# aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

k_list = [len(aspect_ratio)*2 + 2 for aspect_ratio in aspect_ratios]

total_anchors = 0
for (size, k) in zip(feat_size, k_list):
    total_anchors += size*size*k

default_boxes = DefaultBoxes(fig_size, feat_size, steps, scales, aspect_ratios)
