import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import os
from torchsummary import summary
from torch import optim
import torchvision
import torchvision.transforms as transforms


class SEBlock(nn.Module):
    def __init__(self, in_channels, r=0.25):
        super().__init__()

        # se_channels : reduce layer out channels 계산
        se_channels = max(1, int(in_channels*r))

        self.se = nn.Sequential(  # squeeze
            nn.AdaptiveAvgPool2d(1),
            # excitation
            nn.Conv2d(in_channels, se_channels,
                      kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(se_channels, in_channels,
                      kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.se(x)


class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expand, kernel_size, stride=1, r=0.25, dropout_rate=0.2, bias=True):
        super().__init__()

        # 변수 설정
        self.dropout_rate = dropout_rate
        self.expand = expand

        # skip connection 사용을 위한 조건 지정
        self.use_residual = in_channels == out_channels and stride == 1

        # 논문에서 수행한 BatchNorm, SiLU 적용
        # stage1. Expansion
        expand_channels = in_channels*expand
        self.expansion = nn.Sequential(nn.Conv2d(in_channels, expand_channels, 1, bias=False),
                                       nn.BatchNorm2d(
                                           expand_channels, momentum=0.99),
                                       nn.SiLU(),
                                       )

        # stage2. Depth-wise convolution
        self.depth_wise = nn.Sequential(nn.Conv2d(expand_channels, expand_channels, kernel_size=kernel_size, stride=1, padding=1, groups=expand_channels),
                                        nn.BatchNorm2d(
                                            expand_channels, momentum=0.99),
                                        nn.SiLU(),
                                        )

        # stage3. Squeeze and Excitation
        self.se_block = SEBlock(expand_channels, r)

        # stage4. Point-wise convolution
        self.point_wise = nn.Sequential(nn.Conv2d(expand_channels, out_channels, 1, 1, bias=False),
                                        nn.BatchNorm2d(
                                            out_channels, momentum=0.99)
                                        )

    def forward(self, x):

        # stage1
        if self.expand != 1:
            x = self.expansion(x)

        # stage2
        x = self.depth_wise(x)

        # stage3
        x = self.se_block(x)

        # stage4
        x = self.point_wise(x)

        # stage5 skip connection
        res = x

        if self.use_residual:
            if self.training and (self.dropout_rate is not None):
                x = F.dropout2d(input=x, p=self.dropout_rate,
                                training=self.training, inplace=True)

            x = x + res

        return x


class EfficientNet_b7(nn.Module):
    def __init__(self, width = 2.0, depth = 3.1, resolution=600, dropout = 0.5 ,num_classes = 6):
        super().__init__()

        # stage1
        out_ch = int(32*width)
        self.stage1 = nn.Sequential(nn.Conv2d(in_channels=3, out_channels=out_ch, kernel_size=3, stride=2, padding=1),
                                    nn.BatchNorm2d(out_ch, momentum=0.99))

        # stage2
        self.stage2 = nn.Sequential(MBConvBlock(
            in_channels=out_ch, out_channels=16, expand=1, kernel_size=3, stride=1, dropout_rate=dropout))

        # stage3
        self.stage3 = nn.Sequential(MBConvBlock(in_channels=16, out_channels=24, expand=6, kernel_size=3, stride=2, dropout_rate=dropout),
                                    MBConvBlock(in_channels=24, out_channels=24, expand=6,
                                                kernel_size=3, stride=1, dropout_rate=dropout),
                                    )

        # stage4
        self.stage4 = nn.Sequential(MBConvBlock(in_channels=24, out_channels=40, expand=6, kernel_size=5, stride=2, dropout_rate=dropout),
                                    MBConvBlock(in_channels=40, out_channels=40, expand=6,
                                                kernel_size=5, stride=1, dropout_rate=dropout),
                                    )

        # stage5
        self.stage5 = nn.Sequential(MBConvBlock(in_channels=40, out_channels=80, expand=6, kernel_size=3, stride=2, dropout_rate=dropout),
                                    MBConvBlock(in_channels=80, out_channels=80, expand=6,
                                                kernel_size=3, stride=1, dropout_rate=dropout),
                                    MBConvBlock(in_channels=80, out_channels=80, expand=6,
                                                kernel_size=3, stride=1, dropout_rate=dropout),
                                    )

        # stage6
        self.stage6 = nn.Sequential(MBConvBlock(in_channels=80, out_channels=112, expand=6, kernel_size=5, stride=1, dropout_rate=dropout),
                                    MBConvBlock(in_channels=112, out_channels=112, expand=6,
                                                kernel_size=5, stride=1, dropout_rate=dropout),
                                    MBConvBlock(in_channels=112, out_channels=112, expand=6,
                                                kernel_size=5, stride=1, dropout_rate=dropout),
                                    )

        # stage7
        self.stage7 = nn.Sequential(MBConvBlock(in_channels=112, out_channels=192, expand=6, kernel_size=5, stride=2, dropout_rate=dropout),
                                    MBConvBlock(in_channels=192, out_channels=192, expand=6,
                                                kernel_size=5, stride=1, dropout_rate=dropout),
                                    MBConvBlock(in_channels=192, out_channels=192, expand=6,
                                                kernel_size=5, stride=1, dropout_rate=dropout),
                                    MBConvBlock(in_channels=192, out_channels=192, expand=6,
                                                kernel_size=5, stride=1, dropout_rate=dropout),
                                    )

        # stage8
        self.stage8 = nn.Sequential(MBConvBlock(
            in_channels=192, out_channels=320, expand=6, kernel_size=3, stride=1, dropout_rate=dropout))

        # stage9
        self.last_channels = math.ceil(1280*width)
        self.stage9 = nn.Conv2d(
            in_channels=320, out_channels=self.last_channels, kernel_size=1)

        # result
        self.out_layer = nn.Linear(self.last_channels, num_classes)

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        x = self.stage5(x)
        x = self.stage6(x)
        x = self.stage7(x)
        x = self.stage8(x)
        x = self.stage9(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).view(-1, self.last_channels)
        x = self.out_layer(x)

        return x
