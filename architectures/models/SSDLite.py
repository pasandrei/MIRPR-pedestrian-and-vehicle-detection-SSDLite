import torch
import torch.nn as nn

from architectures.backbones.MobileNet import ConvBNReLU, mobilenet_v2

# this adapts the standard resnet implementation from
# https://github.com/NVIDIA/DeepLearningExamples/tree/master/PyTorch/Detection/SSD
# to have depth wise separable convolutions and a mobilenet backbone


class DepthWiseConv_No_ReLu(nn.Module):
    """
    Depth wise convolution used for the final bbox offset and class score predictions
    """
    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0):
        super().__init__()
        self.ds_conv = nn.Conv2d(in_planes, in_planes, kernel_size, groups=in_planes, padding=padding)
        self.ds_bn = nn.BatchNorm2d(in_planes)
        self.pw_conv = nn.Conv2d(in_planes, out_planes, kernel_size=1)

    def forward(self, x):
        return self.pw_conv(self.ds_bn(self.ds_conv(x)))


class DepthWiseConv(nn.Module):
    """
    depth wise followed by point wise convolution
    """

    def __init__(self, in_planes, out_planes, kernel_size=3, stride=1, padding=0, bias=False):
        super().__init__()
        self.ds_conv = ConvBNReLU(in_planes, in_planes, kernel_size=kernel_size,
                                  stride=stride, groups=in_planes, padding=padding, bias=False)
        self.pw_conv = ConvBNReLU(in_planes, out_planes, kernel_size=1, bias=False)

    def forward(self, x):
        return self.pw_conv(self.ds_conv(x))


class SSD_Head(nn.Module):
    def __init__(self, n_classes=81, k_list=[4, 6, 6, 6, 6, 6],
                 out_channels=[576, 1280, 512, 256, 256, 128], width_mult=1):
        super().__init__()
        self.backbone = mobilenet_v2(pretrained=True, width_mult=width_mult, num_classes=n_classes)
        self.out_channels = out_channels

        self.label_num = n_classes
        self._build_additional_features(self.out_channels[1:-1])
        self.num_defaults = k_list
        self.loc = []
        self.conf = []

        for nd, oc in zip(self.num_defaults, self.out_channels):
            self.loc.append(DepthWiseConv_No_ReLu(oc, nd * 4, kernel_size=3, padding=1))
            self.conf.append(DepthWiseConv_No_ReLu(
                oc, nd * self.label_num, kernel_size=3, padding=1))

        self.loc = nn.ModuleList(self.loc)
        self.conf = nn.ModuleList(self.conf)
        self._init_weights()

    def _build_additional_features(self, input_sizes):
        self.additional_blocks = []
        for i, (input_size, output_size) in enumerate(zip(input_sizes[:-1], input_sizes[1:])):
            layer = DepthWiseConv(input_size, output_size, kernel_size=3,
                                  padding=1, stride=2)
            self.additional_blocks.append(layer)

        self.additional_blocks.append(DepthWiseConv(self.out_channels[-2], self.out_channels[-1],
                                                    kernel_size=2))

        self.additional_blocks = nn.ModuleList(self.additional_blocks)

    def _init_weights(self):
        layers = [*self.additional_blocks, *self.loc, *self.conf]
        for layer in layers:
            for param in layer.parameters():
                if param.dim() > 1:
                    nn.init.xavier_uniform_(param)

    # Shape the classifier to the view of bboxes
    def bbox_view(self, src, loc, conf):
        ret = []
        for s, l, c in zip(src, loc, conf):
            ret.append((l(s).view(s.size(0), 4, -1), c(s).view(s.size(0), self.label_num, -1)))

        locs, confs = list(zip(*ret))
        locs, confs = torch.cat(locs, 2).contiguous(), torch.cat(confs, 2).contiguous()
        return locs, confs

    def forward(self, x):
        inter_layer, x = self.backbone(x)
        detection_feed = [inter_layer, x]
        for l in self.additional_blocks:
            x = l(x)
            detection_feed.append(x)

        # Feature Maps 19x19, 10x10, 5x5, 3x3, 1x1
        locs, confs = self.bbox_view(detection_feed, self.loc, self.conf)

        return locs, confs
