import math
from typing import Tuple
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange
from torch import FloatTensor, LongTensor

from .pos_enc import ImgPosEnc


from torch.nn import functional as F
from collections import OrderedDict

from torch import nn
import torch
import math


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# Swish: Swish Activation Function
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x.mul_(x.sigmoid()) if self.inplace else x.mul(x.sigmoid())


class ConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1,
                 groups=1, dilate=1):

        super(ConvBlock, self).__init__()
        dilate = 1 if stride > 1 else dilate
        padding = ((kernel_size - 1) // 2) * dilate

        self.conv_block = nn.Sequential(OrderedDict([
           ("conv", nn.Conv2d(in_channels=in_planes, out_channels=out_planes,
                              kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilate, groups=groups, bias=False)),
            ("norm", nn.BatchNorm2d(num_features=out_planes,
                                    eps=1e-3, momentum=0.01)),
            ("act", Swish(inplace=True))
        ]))

    def forward(self, x):
        return self.conv_block(x)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# SEBlock: Squeeze & Excitation (SCSE)
#          namely, Channel-wise Attention
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class SEBlock(nn.Module):
    def __init__(self, in_planes, reduced_dim):
        super(SEBlock, self).__init__()
        self.channel_se = nn.Sequential(OrderedDict([
            ("linear1", nn.Conv2d(in_planes, reduced_dim, kernel_size=1, stride=1, padding=0, bias=True)),
            ("act", Swish(inplace=True)),
            ("linear2", nn.Conv2d(reduced_dim, in_planes, kernel_size=1, stride=1, padding=0, bias=True))
        ]))

    def forward(self, x):
        x_se = torch.sigmoid(self.channel_se(F.adaptive_avg_pool2d(x, output_size=(1, 1))))
        return torch.mul(x, x_se)


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# MBConvBlock: MBConvBlock for EfficientNet
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class MBConvBlock(nn.Module):
    def __init__(self, in_planes, out_planes,
                 expand_ratio,  kernel_size, stride, dilate,
                 reduction_ratio=4, dropout_rate=0.2):
        super(MBConvBlock, self).__init__()
        self.dropout_rate = dropout_rate
        self.expand_ratio = expand_ratio
        self.use_se = (reduction_ratio is not None) and (reduction_ratio > 1)
        self.use_residual = in_planes == out_planes and stride == 1

        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        dilate = 1 if stride > 1 else dilate
        hidden_dim = in_planes * expand_ratio
        reduced_dim = max(1, int(in_planes / reduction_ratio))

        # step 1. Expansion phase/Point-wise convolution
        if expand_ratio != 1:
            self.expansion = ConvBlock(in_planes, hidden_dim, 1)

        # step 2. Depth-wise convolution phase
        self.depth_wise = ConvBlock(hidden_dim, hidden_dim, kernel_size,
                                    stride=stride, groups=hidden_dim, dilate=dilate)
        # step 3. Squeeze and Excitation
        if self.use_se:
            self.se_block = SEBlock(hidden_dim, reduced_dim)

        # step 4. Point-wise convolution phase
        self.point_wise = nn.Sequential(OrderedDict([
            ("conv", nn.Conv2d(in_channels=hidden_dim,
                               out_channels=out_planes, kernel_size=1,
                               stride=1, padding=0, dilation=1, groups=1, bias=False)),
            ("norm", nn.BatchNorm2d(out_planes, eps=1e-3, momentum=0.01))
        ]))

    def forward(self, x):
        res = x

        # step 1. Expansion phase/Point-wise convolution
        if self.expand_ratio != 1:
            x = self.expansion(x)

        # step 2. Depth-wise convolution phase
        x = self.depth_wise(x)

        # step 3. Squeeze and Excitation
        if self.use_se:
            x = self.se_block(x)

        # step 4. Point-wise convolution phase
        x = self.point_wise(x)

        # step 5. Skip connection and drop connect
        if self.use_residual:
            if self.training and (self.dropout_rate is not None):
                x = F.dropout2d(input=x, p=self.dropout_rate,
                                training=self.training, inplace=True)
            x = x + res

        return x


# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
# EfficientNet: EfficientNet Implementation
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++ #
class EfficientNet(nn.Module):
    def __init__(self, arch="bo", num_classes=1000):
        super(EfficientNet, self).__init__()

        arch_params = {
            # arch width_multi depth_multi input_h dropout_rate
            'b0': (1.0, 1.0, 224, 0.2),
            'b1': (1.0, 1.1, 240, 0.2),
            'b2': (1.1, 1.2, 260, 0.3),
            'b3': (1.2, 1.4, 300, 0.3),
            'b4': (1.4, 1.8, 380, 0.4),
            'b5': (1.6, 2.2, 456, 0.4),
            'b6': (1.8, 2.6, 528, 0.5),
            'b7': (2.0, 3.1, 600, 0.5),
        }
        width_multi, depth_multi, net_h, dropout_rate = arch_params[arch]

        settings = [
            # t, c,  n, k, s, d
            [1, 16, 1, 3, 1, 1],   # 3x3, 112 -> 112
            [6, 24, 2, 3, 2, 1],   # 3x3, 112 ->  56
            [6, 40, 2, 5, 2, 1],   # 5x5, 56  ->  28
            [6, 80, 3, 3, 2, 1],   # 3x3, 28  ->  14
            [6, 112, 3, 5, 1, 1],  # 5x5, 14  ->  14
            [6, 192, 4, 5, 2, 1],  # 5x5, 14  ->   7
            [6, 320, 1, 3, 1, 1],  # 3x3, 7   ->   7
        ]
        self.dropout_rate = dropout_rate
        out_channels = self._round_filters(32, width_multi)
        self.mod1 = ConvBlock(3, out_channels, kernel_size=3, stride=2, groups=1, dilate=1)

        in_channels = out_channels
        drop_rate = self.dropout_rate
        mod_id = 0
        for t, c, n, k, s, d in settings:
            out_channels = self._round_filters(c, width_multi)
            repeats = self._round_repeats(n, depth_multi)

            if self.dropout_rate:
                drop_rate = self.dropout_rate * float(mod_id+1) / len(settings)

            # Create blocks for module
            blocks = []
            for block_id in range(repeats):
                stride = s if block_id == 0 else 1
                dilate = d if stride == 1 else 1

                blocks.append(("block%d" % (block_id + 1), MBConvBlock(in_channels, out_channels,
                                                                       expand_ratio=t, kernel_size=k,
                                                                       stride=stride, dilate=dilate,
                                                                       dropout_rate=drop_rate)))

                in_channels = out_channels
            self.add_module("mod%d" % (mod_id + 2), nn.Sequential(OrderedDict(blocks)))
            mod_id += 1

        self.last_channels = self._round_filters(1280, width_multi)
        self.last_feat = ConvBlock(in_channels, self.last_channels, 1)




    @staticmethod
    def _make_divisible(value, divisor=8):
        new_value = max(divisor, int(value + divisor / 2) // divisor * divisor)
        if new_value < 0.9 * value:
            new_value += divisor
        return new_value

    def _round_filters(self, filters, width_multi):
        if width_multi == 1.0:
            return filters
        return int(self._make_divisible(filters * width_multi))

    @staticmethod
    def _round_repeats(repeats, depth_multi):
        if depth_multi == 1.0:
            return repeats
        return int(math.ceil(depth_multi * repeats))

    def forward(self, x,x_mask):
        x = self.mod2(self.mod1(x))   # (N, 16,   H/2,  W/2)
        x = self.mod3(x)              # (N, 24,   H/4,  W/4)
        x = self.mod4(x)              # (N, 32,   H/8,  W/8)
        x = self.mod6(self.mod5(x))   # (N, 96,   H/16, W/16)
        x = self.mod8(self.mod7(x))   # (N, 320,  H/32, W/32)
        x = self.last_feat(x)

        x = F.adaptive_avg_pool2d(x, (1, 1)).view(-1, self.last_channels)
        if self.training and (self.dropout_rate is not None):
            x = F.dropout(input=x, p=self.dropout_rate,
                          training=self.training, inplace=True)
        out_mask = x_mask[:, 0::2, 0::2]
        out_mask = out_mask[:, 0::2, 0::2]
        out_mask = out_mask[:, 0::2, 0::2]
        out_mask = out_mask[:, 0::2, 0::2]
        return x,out_mask



class Encoder(pl.LightningModule):
    def __init__(self, d_model: int, growth_rate: int, num_layers: int):
        super().__init__()
        arch = "b0"
        img_preparam = {"b0": (224, 0.875),
                        "b1": (240, 0.882),
                        "b2": (260, 0.890),
                        "b3": (300, 0.904),
                        "b4": (380, 0.922),
                        "b5": (456, 0.934),
                        "b6": (528, 0.942),
                        "b7": (600, 0.949)}
        net_h = img_preparam[arch][0]
        self.model = EfficientNet(arch=arch)

        self.feature_proj = nn.Conv2d(1280, d_model, kernel_size=1)

        self.pos_enc_2d = ImgPosEnc(d_model, normalize=True)

        self.norm = nn.LayerNorm(d_model)

    def forward(
        self, img: FloatTensor, img_mask: LongTensor
    ) -> Tuple[FloatTensor, LongTensor]:
        """encode image to feature

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h', w']
        img_mask: LongTensor
            [b, h', w']

        Returns
        -------
        Tuple[FloatTensor, LongTensor]
            [b, h, w, d], [b, h, w]
        """
        # extract feature
        feature, mask = self.model(img, img_mask)
        feature = self.feature_proj(feature)

        # proj
        feature = rearrange(feature, "b d h w -> b h w d")

        # positional encoding
        feature = self.pos_enc_2d(feature, mask)
        feature = self.norm(feature)

        # flat to 1-D
        return feature, mask
