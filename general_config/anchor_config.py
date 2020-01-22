from collections import OrderedDict
from math import sqrt

zoom = {
    20: [1.],
    10: [1.],
    5: [1.],
    3: [1.],
    1: [1.]
}
zoom = OrderedDict(zoom)

sqrt2 = sqrt(2)
sqrt3 = sqrt(3)

ratio = {
    20: [(1., 1.), (1/sqrt2, 1*sqrt2), (1*sqrt2, 1/sqrt2), (1/sqrt3, 1*sqrt3), (1*sqrt3, 1/sqrt3), (1.3, 1.3)],
    10: [(1., 1.), (1/sqrt2, 1*sqrt2), (1*sqrt2, 1/sqrt2), (1/sqrt3, 1*sqrt3), (1*sqrt3, 1/sqrt3), (1.3, 1.3)],
    5: [(1., 1.), (1/sqrt2, 1*sqrt2), (1*sqrt2, 1/sqrt2), (1/sqrt3, 1*sqrt3), (1*sqrt3, 1/sqrt3), (1.3, 1.3)],
    3: [(1., 1.), (1/sqrt2, 1*sqrt2), (1*sqrt2, 1/sqrt2), (1.3, 1.3)],
    1: [(1., 1.), (1/sqrt2, 1*sqrt2), (1*sqrt2, 1/sqrt2), (1.3, 1.3)]
}
ratio = OrderedDict(ratio)

# anchors per feature map cell for each grid
k_list = [len(v_zoom) * len(v_ratio) for (_, v_zoom), (_, v_ratio)
          in zip(zoom.items(), ratio.items())]

total_anchors = sum([len(v_zoom) * len(v_ratio) * k_zoom**2 for (k_zoom, v_zoom), (_, v_ratio)
                     in zip(zoom.items(), ratio.items())])

print("total_anchors: ", total_anchors)
