from architectures.backbones.MobileNet import ConvBNReLU, InvertedResidual, mobilenet_v2
from torch import nn


class OutConv(nn.Module):
    def __init__(self, in_channels, n_classes, k):
        super().__init__()
        self.k = k
        self.oconv1 = nn.Conv2d(in_channels, n_classes*k, 3, padding=1)
        self.oconv2 = nn.Conv2d(in_channels, 4*k, 3, padding=1)

    def forward(self, x):
        return [self.flatten_conv(self.oconv1(x), self.k),
                self.flatten_conv(self.oconv2(x), self.k)]

    def flatten_conv(self, x, k):
        bs, nf, gx, gy = x.size()
        x = x.permute(0, 2, 3, 1).contiguous()
        return x.view(bs, -1, nf//k)


class SSD_Head(nn.Module):
    ''' so far returns just [B x 16 x 4, B x 16 x 3] '''

    def __init__(self, in_channels=1280, n_classes=3, k=1):
        super().__init__()

        # pw to reduce filters
        self.sconv1 = ConvBNReLU(in_channels, 256, 1)
        # reduce dimension
        self.sconv2 = ConvBNReLU(256, 256, 3, stride=2)
        # out
        self.out = OutConv(256, n_classes, k)

        # weight initialization
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

        self.backbone = mobilenet_v2()

        print('Created SSDNet model succesfully!')

    def forward(self, x):
        x = self.backbone(x)
        x = self.sconv1(x)
        x = self.sconv2(x)
        return self.out(x)
