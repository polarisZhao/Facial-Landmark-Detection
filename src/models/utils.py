import torch
from torch import nn
from torch.nn import functional as F


class SpatialPyramidPooling(nn.Module):
    def __init__(self, pool_sizes=[5, 9, 13]):
        super(SpatialPyramidPooling, self).__init__()

        self.maxpools = nn.ModuleList([
            nn.MaxPool2d(pool_size, 1, pool_size // 2)
            for pool_size in pool_sizes
        ])

    def forward(self, x):
        features = [maxpool(x) for maxpool in self.maxpools[::-1]]
        features = torch.cat(features + [x], dim=1)

        return features


# class HS(nn.Module):
#     def __init__(self):
#         super(HS, self).__init__()

#     def forward(self, x):
#         #inlining this saves 1 second per epoch (V100 GPU) vs having a temp x and then returning x(!)
#         return x * (torch.tanh(F.softplus(x)))


class Mish(nn.Module):
    def __init__(self):
        super(Mish, self).__init__()

    def forward(self, x):
        return x * (torch.tanh(F.softplus(x)))


def conv_bn1X1(inp, oup, stride, leaky=0):
    return nn.Sequential(nn.Conv2d(inp, oup, 1, stride, padding=0, bias=False),
                         nn.BatchNorm2d(oup),
                         nn.LeakyReLU(negative_slope=leaky))


def conv_bn(inp, oup, stride=1, leaky=0):
    return nn.Sequential(nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                         nn.BatchNorm2d(oup),
                         nn.LeakyReLU(negative_slope=leaky))


class FPN(nn.Module):
    def __init__(self, in_channels_list, out_channels=98):
        super(FPN, self).__init__()
        leaky = 0
        if (out_channels <= 64):
            leaky = 0.1
        self.output1 = conv_bn1X1(in_channels_list[0],
                                  out_channels,
                                  stride=1,
                                  leaky=leaky)
        self.output2 = conv_bn1X1(in_channels_list[1],
                                  out_channels,
                                  stride=1,
                                  leaky=leaky)
        self.output3 = conv_bn1X1(in_channels_list[2],
                                  out_channels,
                                  stride=1,
                                  leaky=leaky)

        self.merge1 = conv_bn(out_channels, out_channels, leaky=leaky)
        self.merge2 = conv_bn(out_channels, out_channels, leaky=leaky)

    def forward(self, input):
        output1 = self.output1(input[0])
        output2 = self.output2(input[1])
        output3 = self.output3(input[2])

        up3 = F.interpolate(output3,
                            size=[output2.size(2),
                                  output2.size(3)],
                            mode="nearest")
        output2 = output2 + up3
        output2 = self.merge2(output2)

        up2 = F.interpolate(output2,
                            size=[output1.size(2),
                                  output1.size(3)],
                            mode="nearest")
        output1 = output1 + up2
        output1 = self.merge1(output1)

        return [output1, output2, output3]


class SELayer(nn.Module):
    def __init__(self, inplanes, isTensor=True):
        super(SELayer, self).__init__()
        if isTensor:
            # if the input is (N, C, H, W)
            self.SE_opr = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(inplanes,
                          inplanes // 4,
                          kernel_size=1,
                          stride=1,
                          bias=False),
                nn.BatchNorm2d(inplanes // 4),
                nn.ReLU(inplace=True),
                nn.Conv2d(inplanes // 4,
                          inplanes,
                          kernel_size=1,
                          stride=1,
                          bias=False),
            )
        else:
            # if the input is (N, C)
            self.SE_opr = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Linear(inplanes, inplanes // 4, bias=False),
                nn.BatchNorm1d(inplanes // 4),
                nn.ReLU(inplace=True),
                nn.Linear(inplanes // 4, inplanes, bias=False),
            )

    def forward(self, x):
        atten = self.SE_opr(x)
        atten = torch.clamp(atten + 3, 0, 6) / 6
        return x * atten