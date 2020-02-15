from collections import OrderedDict
from math import sqrt

zoom = {
    19: [1.],
    10: [1.],
    5: [1.],
    3: [1.],
    2: [1.],
    1: [1.]
}
zoom = OrderedDict(zoom)

sqrt2 = sqrt(2)
sqrt3 = sqrt(3)

ratio = {
    19: [(2.85, 2.85), (2.85/sqrt2, 2.85*sqrt2), (2.85*sqrt2, 2.85/sqrt2)],
    10: [(3.3, 3.3), (3.3/sqrt2, 3.3*sqrt2), (3.3*sqrt2, 3.3/sqrt2), (3.3/sqrt3, 3.3*sqrt3), (3.3*sqrt3, 3.3/sqrt3), (4.1, 4.1)],
    5: [(2.55, 2.55), (2.55/sqrt2, 2.55*sqrt2), (2.55*sqrt2, 2.55/sqrt2), (2.55/sqrt3, 2.55*sqrt3), (2.55*sqrt3, 2.55/sqrt3), (3, 3)],
    3: [(2.07, 2.07), (2.07/sqrt2, 2.07*sqrt2), (2.07*sqrt2, 2.07/sqrt2), (2.3, 2.3)],
    2: [(2.07, 2.07), (2.07/sqrt2, 2.07*sqrt2), (2.07*sqrt2, 2.07/sqrt2), (2.3, 2.3)],
    1: [(0.87, 0.87), (0.87/sqrt2, 0.87*sqrt2), (0.87*sqrt2, 0.87/sqrt2), (0.95, 0.95)]
}
ratio = OrderedDict(ratio)

# anchors per feature map cell for each grid
k_list = [len(v_zoom) * len(v_ratio) for (_, v_zoom), (_, v_ratio)
          in zip(zoom.items(), ratio.items())]

total_anchors = sum([len(v_zoom) * len(v_ratio) * k_zoom**2 for (k_zoom, v_zoom), (_, v_ratio)
                     in zip(zoom.items(), ratio.items())])
