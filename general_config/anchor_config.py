from collections import OrderedDict

zoom = {
    20: [1., 1.25, 1.5, 1.75],
    10: [1., 1.25, 1.5, 1.75],
    5: [0.75, 1., 1.25, 1.5],
    3: [0.75, 1., 1.25, 1.5],
    2: [0.75, 1., 1.25, 1.5],
    1: [0.75, 1., 1.25, 1.5]
}
zoom = OrderedDict(zoom)

ratio = {
    20: [(1., 1.)],
    10: [(1., 1.), (1., 0.5), (0.5, 1), (2, 1), (1, 2), (1, 0.25)],
    5: [(1., 1.), (1., 0.5), (0.5, 1), (2, 1), (1, 2), (1, 0.25)],
    3: [(1., 1.), (1., 0.5), (0.5, 1), (2, 1), (1, 2), (1, 0.25)],
    2: [(1., 1.), (1., 0.5), (0.5, 1), (2, 1), (1, 2), (1, 0.25)],
    1: [(1., 1.), (1., 0.5), (0.5, 1), (2, 1), (1, 2), (1, 0.25)]
}
ratio = OrderedDict(ratio)

# anchors per feature map cell for each grid
k_list = [len(v_zoom) * len(v_ratio) for (_, v_zoom), (_, v_ratio)
         in zip(zoom.items(), ratio.items())]

total_anchors = sum([len(v_zoom) * len(v_ratio) * k_zoom**2 for (k_zoom, v_zoom), (_, v_ratio)
         in zip(zoom.items(), ratio.items())])
