"""Inference-only large LaMa generator derived from BallonsTranslator.

Source: https://github.com/dmMaze/BallonsTranslator, files
ballontranslator/modules/inpaint/lama.py and ffc.py. GPL-3.0; see
LICENSE-BallonsTranslator. FFC originally comes from pkumivision/FFC.

This is the 18-block, sigmoid-output, no-MPE, no-LFU checkpoint architecture.
Training, discriminators and unused positional encodings are omitted. Parameter
names match gen_state_dict. FFTs retain their complex values throughout; the
adapter selects CPU for backends that cannot execute them correctly.
"""

import torch
from torch import nn


class FourierUnit(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_layer = nn.Conv2d(in_channels * 2, out_channels * 2, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        batch, _, height, width = x.shape
        ffted = torch.fft.rfftn(x, dim=(-2, -1), norm='ortho')
        ffted = torch.stack((ffted.real, ffted.imag), dim=-1)
        ffted = ffted.permute(0, 1, 4, 2, 3).contiguous()
        ffted = ffted.view(batch, -1, height, width // 2 + 1)
        ffted = self.relu(self.bn(self.conv_layer(ffted)))
        ffted = ffted.view(batch, -1, 2, height, width // 2 + 1)
        ffted = ffted.permute(0, 1, 3, 4, 2).contiguous()
        spectrum = torch.complex(ffted[..., 0], ffted[..., 1])
        return torch.fft.irfftn(spectrum, s=(height, width), dim=(-2, -1), norm='ortho')


class SpectralTransform(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.downsample = nn.AvgPool2d(2, stride=2) if stride == 2 else nn.Identity()
        self.conv1 = nn.Sequential(nn.Conv2d(in_channels, out_channels // 2, 1, bias=False),
                                   nn.BatchNorm2d(out_channels // 2), nn.ReLU(inplace=True))
        self.fu = FourierUnit(out_channels // 2, out_channels // 2)
        self.conv2 = nn.Conv2d(out_channels // 2, out_channels, 1, bias=False)

    def forward(self, x):
        x = self.conv1(self.downsample(x))
        return self.conv2(x + self.fu(x))


class FFC(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0):
        super().__init__()
        in_g, out_g = int(in_channels * ratio_gin), int(out_channels * ratio_gout)
        in_l, out_l = in_channels - in_g, out_channels - out_g
        self.ratio_gout = ratio_gout

        def conv(cin, cout):
            return (nn.Conv2d(cin, cout, kernel_size, stride, padding,
                              bias=False, padding_mode='reflect')
                    if cin and cout else nn.Identity())

        self.convl2l = conv(in_l, out_l)
        self.convl2g = conv(in_l, out_g)
        self.convg2l = conv(in_g, out_l)
        self.convg2g = SpectralTransform(in_g, out_g, stride) if in_g and out_g else nn.Identity()

    def forward(self, x):
        local, global_ = x if isinstance(x, tuple) else (x, 0)
        out_l = self.convl2l(local) + self.convg2l(global_) if self.ratio_gout != 1 else 0
        out_g = self.convl2g(local) + self.convg2g(global_) if self.ratio_gout != 0 else 0
        return out_l, out_g


class FFC_BN_ACT(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, ratio_gin, ratio_gout,
                 stride=1, padding=0):
        super().__init__()
        self.ffc = FFC(in_channels, out_channels, kernel_size, ratio_gin, ratio_gout,
                       stride=stride, padding=padding)
        global_channels = int(out_channels * ratio_gout)
        self.bn_l = nn.BatchNorm2d(out_channels - global_channels) if ratio_gout != 1 else nn.Identity()
        self.bn_g = nn.BatchNorm2d(global_channels) if ratio_gout != 0 else nn.Identity()
        self.act_l = nn.ReLU(inplace=True) if ratio_gout != 1 else nn.Identity()
        self.act_g = nn.ReLU(inplace=True) if ratio_gout != 0 else nn.Identity()

    def forward(self, x):
        local, global_ = self.ffc(x)
        return self.act_l(self.bn_l(local)), self.act_g(self.bn_g(global_))


class FFCResnetBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv1 = FFC_BN_ACT(channels, channels, 3, .75, .75, padding=1)
        self.conv2 = FFC_BN_ACT(channels, channels, 3, .75, .75, padding=1)

    def forward(self, x):
        local, global_ = self.conv2(self.conv1(x))
        return x[0] + local, x[1] + global_


class ConcatTupleLayer(nn.Module):
    def forward(self, x):
        return torch.cat(x, dim=1)


class LamaLargeGenerator(nn.Module):
    def __init__(self):
        super().__init__()
        layers = [nn.ReflectionPad2d(3), FFC_BN_ACT(4, 64, 7, 0, 0)]
        for cin, cout, ratio in ((64, 128, 0), (128, 256, 0), (256, 512, .75)):
            layers.append(FFC_BN_ACT(cin, cout, 3, 0, ratio, stride=2, padding=1))
        layers.extend(FFCResnetBlock(512) for _ in range(18))
        layers.append(ConcatTupleLayer())
        for cin, cout in ((512, 256), (256, 128), (128, 64)):
            layers.extend([nn.ConvTranspose2d(cin, cout, 3, stride=2, padding=1, output_padding=1),
                           nn.BatchNorm2d(cout), nn.ReLU(inplace=True)])
        layers.extend([nn.ReflectionPad2d(3), nn.Conv2d(64, 3, 7), nn.Sigmoid()])
        self.model = nn.Sequential(*layers)

    def forward(self, image, mask):
        return self.model(torch.cat((image * (1 - mask), mask), dim=1))
