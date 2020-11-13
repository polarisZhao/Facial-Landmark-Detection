import torch
import torch.nn as nn
from torch.nn import functional as F

from .utils import FPN, Mish, SELayer


class Shufflenet(nn.Module):
    def __init__(self, inp, oup, base_mid_channels, *, ksize, stride,
                 activation, useSE):
        super(Shufflenet, self).__init__()
        self.stride = stride
        assert stride in [1, 2]
        assert ksize in [3, 5, 7]
        assert base_mid_channels == oup // 2

        self.base_mid_channel = base_mid_channels
        self.ksize = ksize
        pad = ksize // 2
        self.pad = pad
        self.inp = inp

        outputs = oup - inp

        branch_main = [
            # pw
            nn.Conv2d(inp, base_mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(base_mid_channels),
            None,
            # dw
            nn.Conv2d(base_mid_channels,
                      base_mid_channels,
                      ksize,
                      stride,
                      pad,
                      groups=base_mid_channels,
                      bias=False),
            nn.BatchNorm2d(base_mid_channels),
            # pw-linear
            nn.Conv2d(base_mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            None,
        ]
        if activation == 'ReLU':
            assert useSE == False
            '''This model should not have SE with ReLU'''
            branch_main[2] = nn.ReLU(inplace=True)
            branch_main[-1] = nn.ReLU(inplace=True)
        else:
            branch_main[2] = Mish()
            branch_main[-1] = Mish()
            if useSE:
                branch_main.append(SELayer(outputs))
        self.branch_main = nn.Sequential(*branch_main)

        if stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, ksize, stride, pad, groups=inp,
                          bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                None,
            ]
            if activation == 'ReLU':
                branch_proj[-1] = nn.ReLU(inplace=True)
            else:
                branch_proj[-1] = Mish()
            self.branch_proj = nn.Sequential(*branch_proj)
        else:
            self.branch_proj = None

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)),
                             1)


class Shuffle_Xception(nn.Module):
    def __init__(self, inp, oup, base_mid_channels, *, stride, activation,
                 useSE):
        super(Shuffle_Xception, self).__init__()

        assert stride in [1, 2]
        assert base_mid_channels == oup // 2

        self.base_mid_channel = base_mid_channels
        self.stride = stride
        self.ksize = 3
        self.pad = 1
        self.inp = inp
        outputs = oup - inp

        branch_main = [
            # dw
            nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
            nn.BatchNorm2d(inp),
            # pw
            nn.Conv2d(inp, base_mid_channels, 1, 1, 0, bias=False),
            nn.BatchNorm2d(base_mid_channels),
            None,
            # dw
            nn.Conv2d(base_mid_channels,
                      base_mid_channels,
                      3,
                      stride,
                      1,
                      groups=base_mid_channels,
                      bias=False),
            nn.BatchNorm2d(base_mid_channels),
            # pw
            nn.Conv2d(base_mid_channels,
                      base_mid_channels,
                      1,
                      1,
                      0,
                      bias=False),
            nn.BatchNorm2d(base_mid_channels),
            None,
            # dw
            nn.Conv2d(base_mid_channels,
                      base_mid_channels,
                      3,
                      stride,
                      1,
                      groups=base_mid_channels,
                      bias=False),
            nn.BatchNorm2d(base_mid_channels),
            # pw
            nn.Conv2d(base_mid_channels, outputs, 1, 1, 0, bias=False),
            nn.BatchNorm2d(outputs),
            None,
        ]

        if activation == 'ReLU':
            branch_main[4] = nn.ReLU(inplace=True)
            branch_main[9] = nn.ReLU(inplace=True)
            branch_main[14] = nn.ReLU(inplace=True)
        else:
            branch_main[4] = Mish()
            branch_main[9] = Mish()
            branch_main[14] = Mish()
        assert None not in branch_main

        if useSE:
            assert activation != 'ReLU'
            branch_main.append(SELayer(outputs))

        self.branch_main = nn.Sequential(*branch_main)

        if self.stride == 2:
            branch_proj = [
                # dw
                nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                nn.BatchNorm2d(inp),
                # pw-linear
                nn.Conv2d(inp, inp, 1, 1, 0, bias=False),
                nn.BatchNorm2d(inp),
                None,
            ]
            if activation == 'ReLU':
                branch_proj[-1] = nn.ReLU(inplace=True)
            else:
                branch_proj[-1] = Mish()
            self.branch_proj = nn.Sequential(*branch_proj)

    def forward(self, old_x):
        if self.stride == 1:
            x_proj, x = channel_shuffle(old_x)
            return torch.cat((x_proj, self.branch_main(x)), 1)
        elif self.stride == 2:
            x_proj = old_x
            x = old_x
            return torch.cat((self.branch_proj(x_proj), self.branch_main(x)),
                             1)


def channel_shuffle(x):
    batchsize, num_channels, height, width = x.data.size()
    assert (num_channels % 4 == 0)
    x = x.reshape(batchsize * num_channels // 2, 2, height * width)
    x = x.permute(1, 0, 2)
    x = x.reshape(2, -1, num_channels // 2, height, width)
    return x[0], x[1]


class ShuffleNetV2_Plus(nn.Module):
    def __init__(self,
                 input_size=112,
                 n_class=1000,
                 architecture=None,
                 model_size='Large'):
        super(ShuffleNetV2_Plus, self).__init__()

        assert input_size % 16 == 0
        assert architecture is not None

        self.stage_repeats = [4, 4, 8, 4]
        if model_size == 'Large':
            self.stage_out_channels = [-1, 16, 68, 168, 336, 672, 1280]
        elif model_size == 'Medium':
            self.stage_out_channels = [-1, 16, 48, 128, 256, 512, 1280]
        elif model_size == 'Small':
            self.stage_out_channels = [-1, 16, 36, 104, 208, 416, 1280]
        else:
            raise NotImplementedError

        # building first layer
        input_channel = self.stage_out_channels[1]
        self.first_conv = nn.Sequential(
            nn.Conv2d(3, input_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(input_channel),
            Mish(),
        )

        # building features layer
        self.features = []
        archIndex = 0
        for idxstage in range(len(self.stage_repeats)):
            numrepeat = self.stage_repeats[idxstage]
            output_channel = self.stage_out_channels[idxstage + 2]

            activation = 'Mish' if idxstage >= 1 else 'ReLU'
            useSE = 'True' if idxstage >= 2 else False

            for i in range(numrepeat):
                if i == 0:
                    inp, outp, stride = input_channel, output_channel, 2
                else:
                    inp, outp, stride = input_channel // 2, output_channel, 1

                blockIndex = architecture[archIndex]
                archIndex += 1
                if blockIndex == 0:
                    #print('Shuffle3x3')
                    self.features.append(
                        Shufflenet(inp,
                                   outp,
                                   base_mid_channels=outp // 2,
                                   ksize=3,
                                   stride=stride,
                                   activation=activation,
                                   useSE=useSE))
                elif blockIndex == 1:
                    # print('Shuffle5x5')
                    self.features.append(
                        Shufflenet(inp,
                                   outp,
                                   base_mid_channels=outp // 2,
                                   ksize=5,
                                   stride=stride,
                                   activation=activation,
                                   useSE=useSE))
                elif blockIndex == 2:
                    # print('Shuffle7x7')
                    self.features.append(
                        Shufflenet(inp,
                                   outp,
                                   base_mid_channels=outp // 2,
                                   ksize=7,
                                   stride=stride,
                                   activation=activation,
                                   useSE=useSE))
                elif blockIndex == 3:
                    # print('Xception')
                    self.features.append(
                        Shuffle_Xception(inp,
                                         outp,
                                         base_mid_channels=outp // 2,
                                         stride=stride,
                                         activation=activation,
                                         useSE=useSE))
                else:
                    raise NotImplementedError
                input_channel = output_channel
        assert archIndex == len(architecture)
        self.stage1 = nn.Sequential(*self.features[:4])
        self.stage2 = nn.Sequential(*self.features[4:8])
        self.stage3 = nn.Sequential(*self.features[8:16])
        self.stage4 = nn.Sequential(*self.features[16:])

        # building last layer
        self.conv_last = nn.Sequential(
            nn.Conv2d(input_channel, 1280, 1, 1, 0, bias=False),
            nn.BatchNorm2d(1280), Mish())

        # self.spp = SpatialPyramidPooling()

        self._initialize_weights()

    def forward(self, x):
        x = self.first_conv(x)
        s1 = self.stage1(x)
        s2 = self.stage2(s1)
        s3 = self.stage3(s2)
        s4 = self.stage4(s3)
        out = self.conv_last(s4)
        return s2, s3, out

    def _initialize_weights(self):
        for name, m in self.named_modules():
            if isinstance(m, nn.Conv2d):
                if 'first' in name or 'SE' in name:
                    nn.init.normal_(m.weight, 0, 0.01)
                else:
                    nn.init.normal_(m.weight, 0, 1.0 / m.weight.shape[1])
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.0001)
                nn.init.constant_(m.running_mean, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class shufflenetModel(nn.Module):
    def __init__(self):
        super(shufflenetModel, self).__init__()
        architecture = [
            0, 0, 3, 1, 1, 1, 0, 0, 2, 0, 2, 1, 1, 0, 2, 0, 2, 1, 3, 2
        ]
        self.backbone = ShuffleNetV2_Plus(architecture=architecture)
        self.fpn = FPN(in_channels_list=[168, 336, 1280])
        self.landmarkhead = nn.Conv2d(98,
                                      98,
                                      kernel_size=(1, 1),
                                      stride=1,
                                      padding=0)

    def forward(self, x):
        s3, s4, s5 = self.backbone(x)
        s3, s4, s5 = self.fpn([s3, s4, s5])
        out = self.landmarkhead(s3)
        return out


# if __name__ == "__main__":
#     model = shufflenetModel()
#     test_data = torch.rand(5, 3, 256, 256)
#     s = model(test_data)
#     print(s.size())
