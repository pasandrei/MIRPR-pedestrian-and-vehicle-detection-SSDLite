from collections import OrderedDict
import math

zoom = {
    20: [1],
    10: [1],
    5: [1],
    3: [1],
    2: [1],
    1: [1]
}
zoom = OrderedDict(zoom)

_sqrt2 = math.sqrt(2)
_sqrt3 = math.sqrt(3)
ratio = {
    20: [(1., 1.), (1.25, 1.25), (_sqrt2, 1/_sqrt2), (1/_sqrt2, _sqrt2), (_sqrt3, 1/_sqrt3), (1/_sqrt3, _sqrt3)],
    10: [(1., 1.), (1.25, 1.25), (_sqrt2, 1/_sqrt2), (1/_sqrt2, _sqrt2), (_sqrt3, 1/_sqrt3), (1/_sqrt3, _sqrt3)],
    5: [(1., 1.), (1.25, 1.25), (_sqrt2, 1/_sqrt2), (1/_sqrt2, _sqrt2), (_sqrt3, 1/_sqrt3), (1/_sqrt3, _sqrt3)],
    3: [(1., 1.), (1.25, 1.25), (_sqrt2, 1/_sqrt2), (1/_sqrt2, _sqrt2), (_sqrt3, 1/_sqrt3), (1/_sqrt3, _sqrt3)],
    2: [(1., 1.), (1.25, 1.25), (_sqrt2, 1/_sqrt2), (1/_sqrt2, _sqrt2), (_sqrt3, 1/_sqrt3), (1/_sqrt3, _sqrt3)],
    1: [(1., 1.), (1.25, 1.25), (_sqrt2, 1/_sqrt2), (1/_sqrt2, _sqrt2), (_sqrt3, 1/_sqrt3), (1/_sqrt3, _sqrt3)]
}
ratio = OrderedDict(ratio)

# anchors per feature map cell for each grid
k_list = [len(v_zoom) * len(v_ratio) for (_, v_zoom), (_, v_ratio)
          in zip(zoom.items(), ratio.items())]

total_anchors = sum([len(v_zoom) * len(v_ratio) * k_zoom**2 for (k_zoom, v_zoom), (_, v_ratio)
                     in zip(zoom.items(), ratio.items())])
