from typing import Any
import torch
from torch import Tensor
from torch import nn


####################################### 모델구조정의 ##########################################
####################################### 모델구조정의 ##########################################
####################################### 모델구조정의 ##########################################
class TH_InceptionV4(nn.Module):

    def __init__(self, k=192, l=224, m=256, n=384, num_classes=7):
        super(TH_InceptionV4, self).__init__()

        self.stem = InceptionV4Stem(3)

        self.inceptionA1 = InceptionA(384)
        self.inceptionA2 = InceptionA(384)
        self.inceptionA3 = InceptionA(384)
        self.inceptionA4 = InceptionA(384)
        self.no_etc_output_linear = nn.Linear(384, 2)

        self.reductionA = ReductionA(384, k, l, m, n)

        self.inceptionB1 = InceptionB(1024)
        self.inceptionB2 = InceptionB(1024)
        self.inceptionB3 = InceptionB(1024)
        self.inceptionB4 = InceptionB(1024)
        self.inceptionB5 = InceptionB(1024)
        self.inceptionB6 = InceptionB(1024)
        self.inceptionB7 = InceptionB(1024)

        self.burr_broken_output_linear = nn.Linear(1024, 2)

        self.reductionB = ReductionB(1024)

        self.inceptionC1_1 = InceptionC(1536)
        self.inceptionC2_1 = InceptionC(1536)
        self.inceptionC3_1 = InceptionC(1536)

        self.inceptionB1_1 = InceptionB(1024)
        self.inceptionB2_1 = InceptionB(1024)
        self.inceptionB3_1 = InceptionB(1024)
        self.inceptionB4_1 = InceptionB(1024)
        self.inceptionB5_1 = InceptionB(1024)
        self.inceptionB6_1 = InceptionB(1024)
        self.inceptionB7_1 = InceptionB(1024)

        self.reductionB_1 = ReductionB(1024)

        self.inceptionC1_1 = InceptionC(1536)
        self.inceptionC2_1 = InceptionC(1536)
        self.inceptionC3_1 = InceptionC(1536)

        self.reductionB_2 = ReductionB(1024)

        self.inceptionC1_2 = InceptionC(1536)
        self.inceptionC2_2 = InceptionC(1536)
        self.inceptionC3_2 = InceptionC(1536)

        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.linear = nn.Linear(1536, 1)

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x):
        outputs = []
        x = self.stem(x)

        x1 = self.inceptionA1(x)
        x2 = self.inceptionA2(x1)
        x3 = self.inceptionA3(x2)
        x4 = self.inceptionA4(x3)

        #no_lens와 etc 클래스 output
        no_etc_output = self.global_average_pooling(x4)
        no_etc_output = torch.flatten(no_etc_output, 1)
        no_etc_output = self.no_etc_output_linear(no_etc_output)
        outputs.append(no_etc_output)

        x_redA = self.reductionA(x4)

        xB1 = self.inceptionB1(x_redA)
        xB2 = self.inceptionB2(xB1)
        xB3 = self.inceptionB3(xB2)
        xB4 = self.inceptionB4(xB3)
        xB5 = self.inceptionB5(xB4)
        xB6 = self.inceptionB6(xB5)
        xB7 = self.inceptionB7(xB6)

        burr_broken_output = self.global_average_pooling(xB7)
        burr_broken_output = torch.flatten(burr_broken_output, 1)
        burr_broken_output = self.burr_broken_output_linear(burr_broken_output)
        outputs.append(burr_broken_output)

        #b_edge분기 - broken에서 받음
        x_redB = self.reductionB_1(xB7)

        xC1 = self.inceptionC1_1(x_redB)
        xC2 = self.inceptionC2_1(xC1)
        xC3 = self.inceptionC3_1(xC2)

        b_edge_output = self.global_average_pooling(xC3)
        b_edge_output = torch.flatten(b_edge_output, 1)
        b_edge_output = self.linear(b_edge_output)
        outputs.append(b_edge_output)

        #b_bubble분기 - 완전히 따로 내려옴
        x_redA_2 = self.reductionA(x4)

        xB1_2 = self.inceptionB1_1(x_redA_2)
        xB2_2 = self.inceptionB2_1(xB1_2)
        xB3_2 = self.inceptionB3_1(xB2_2)
        xB4_2 = self.inceptionB4_1(xB3_2)
        xB5_2 = self.inceptionB5_1(xB4_2)
        xB6_2 = self.inceptionB6_1(xB5_2)
        xB7_2 = self.inceptionB7_1(xB6_2)

        x_redB_2 = self.reductionB_2(xB7_2)

        xC1_2 = self.inceptionC1_2(x_redB_2)
        xC2_2 = self.inceptionC2_2(xC1_2)
        xC3_2 = self.inceptionC3_2(xC2_2)

        b_bubble_output = self.global_average_pooling(xC3_2)
        b_bubble_output = torch.flatten(b_bubble_output, 1)
        b_bubble_output = self.linear(b_bubble_output)
        outputs.append(b_bubble_output)

        final_outputs = torch.cat(outputs, dim=1)
        return final_outputs

    def _initialize_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
                stddev = float(module.stddev) if hasattr(module, "stddev") else 0.1
                torch.nn.init.trunc_normal_(module.weight, mean=0.0, std=stddev, a=-2, b=2)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

class BasicConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, **kwargs: Any) -> None:
        super(BasicConv2d, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)
        self.relu = nn.ReLU(True)

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)

        return out

class InceptionV4Stem(nn.Module):
    def __init__(
            self,
            in_channels: int,
    ) -> None:
        super(InceptionV4Stem, self).__init__()
        self.conv2d_1a_3x3 = BasicConv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))

        self.conv2d_2a_3x3 = BasicConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.conv2d_2b_3x3 = BasicConv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.mixed_3a_branch_0 = nn.MaxPool2d((3, 3), (2, 2))
        self.mixed_3a_branch_1 = BasicConv2d(64, 96, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))

        self.mixed_4a_branch_0 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
        )
        self.mixed_4a_branch_1 = nn.Sequential(
            BasicConv2d(160, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(64, 64, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
            BasicConv2d(64, 64, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
            BasicConv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        )

        self.mixed_5a_branch_0 = BasicConv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        self.mixed_5a_branch_1 = nn.MaxPool2d((3, 3), (2, 2))

    def forward(self, x: Tensor) -> Tensor:
        out = self.conv2d_1a_3x3(x)
        out = self.conv2d_2a_3x3(out)
        out = self.conv2d_2b_3x3(out)

        mixed_3a_branch_0 = self.mixed_3a_branch_0(out)
        mixed_3a_branch_1 = self.mixed_3a_branch_1(out)
        mixed_3a_out = torch.cat([mixed_3a_branch_0, mixed_3a_branch_1], 1)

        mixed_4a_branch_0 = self.mixed_4a_branch_0(mixed_3a_out)
        mixed_4a_branch_1 = self.mixed_4a_branch_1(mixed_3a_out)
        mixed_4a_out = torch.cat([mixed_4a_branch_0, mixed_4a_branch_1], 1)

        mixed_5a_branch_0 = self.mixed_5a_branch_0(mixed_4a_out)
        mixed_5a_branch_1 = self.mixed_5a_branch_1(mixed_4a_out)
        mixed_5a_out = torch.cat([mixed_5a_branch_0, mixed_5a_branch_1], 1)

        return mixed_5a_out

class InceptionV4ResNetStem(nn.Module):
    def __init__(
            self,
            in_channels: int,
    ) -> None:
        super(InceptionV4ResNetStem, self).__init__()
        self.features = nn.Sequential(
            BasicConv2d(in_channels, 32, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
            BasicConv2d(32, 32, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(32, 64, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.MaxPool2d((3, 3), (2, 2)),
            BasicConv2d(64, 80, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(80, 192, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
            nn.MaxPool2d((3, 3), (2, 2)),
        )
        self.branch_0 = BasicConv2d(192, 96, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch_1 = nn.Sequential(
            BasicConv2d(192, 48, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(48, 64, kernel_size=(5, 5), stride=(1, 1), padding=(2, 2)),
        )
        self.branch_2 = nn.Sequential(
            BasicConv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BasicConv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1, count_include_pad=False),
            BasicConv2d(192, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )

    def forward(self, x):
        features = self.features(x)
        branch_0 = self.branch_0(features)
        branch_1 = self.branch_1(features)
        branch_2 = self.branch_2(features)
        branch_3 = self.branch_3(features)

        out = torch.cat([branch_0, branch_1, branch_2, branch_3], 1)

        return out

class InceptionA(nn.Module):
    def __init__(
            self,
            in_channels: int,
    ) -> None:
        super(InceptionA, self).__init__()
        self.branch_0 = BasicConv2d(in_channels, 96, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.branch_2 = nn.Sequential(
            BasicConv2d(in_channels, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(64, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BasicConv2d(96, 96, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.brance_3 = nn.Sequential(
            nn.AvgPool2d((3, 3), (1, 1), (1, 1), count_include_pad=False),
            BasicConv2d(384, 96, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )

    def forward(self, x: Tensor) -> Tensor:
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        brance_3 = self.brance_3(x)

        out = torch.cat([branch_0, branch_1, branch_2, brance_3], 1)

        return out

class ReductionA(nn.Module):
    def __init__(
            self,
            in_channels: int,
            k: int,
            l: int,
            m: int,
            n: int,
    ) -> None:
        super(ReductionA, self).__init__()
        self.branch_0 = BasicConv2d(in_channels, n, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, k, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(k, l, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BasicConv2d(l, m, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
        )
        self.branch_2 = nn.MaxPool2d((3, 3), (2, 2))

    def forward(self, x: Tensor) -> Tensor:
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)

        out = torch.cat([branch_0, branch_1, branch_2], 1)

        return out

class InceptionB(nn.Module):
    def __init__(
            self,
            in_channels: int,
    ) -> None:
        super(InceptionB, self).__init__()
        self.branch_0 = BasicConv2d(in_channels, 384, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(192, 224, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
            BasicConv2d(224, 256, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
        )
        self.branch_2 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(192, 192, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
            BasicConv2d(192, 224, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
            BasicConv2d(224, 224, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
            BasicConv2d(224, 256, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d((3, 3), (1, 1), (1, 1), count_include_pad=False),
            BasicConv2d(in_channels, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
        )

    def forward(self, x: Tensor) -> Tensor:
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)

        out = torch.cat([branch_0, branch_1, branch_2, branch_3], 1)

        return out

class ReductionB(nn.Module):
    def __init__(
            self,
            in_channels: int,
    ) -> None:
        super(ReductionB, self).__init__()
        self.branch_0 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(192, 192, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
        )
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(256, 256, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
            BasicConv2d(256, 320, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
            BasicConv2d(320, 320, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
        )
        self.branch_2 = nn.MaxPool2d((3, 3), (2, 2))

    def forward(self, x: Tensor) -> Tensor:
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)

        out = torch.cat([branch_0, branch_1, branch_2], 1)

        return out

class InceptionC(nn.Module):
    def __init__(
            self,
            in_channels: int,
    ) -> None:
        super(InceptionC, self).__init__()
        self.branch_0 = BasicConv2d(in_channels, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        self.branch_1 = BasicConv2d(in_channels, 384, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch_1_1 = BasicConv2d(384, 256, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.branch_1_2 = BasicConv2d(384, 256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))

        self.branch_2 = nn.Sequential(
            BasicConv2d(in_channels, 384, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(384, 448, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            BasicConv2d(448, 512, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
        )
        self.branch_2_1 = BasicConv2d(512, 256, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.branch_2_2 = BasicConv2d(512, 256, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))

        self.branch_3 = nn.Sequential(
            nn.AvgPool2d((3, 3), (1, 1), (1, 1)),
            BasicConv2d(in_channels, 256, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )

    def forward(self, x: Tensor) -> Tensor:
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)

        branch_1_1 = self.branch_1_1(branch_1)
        branch_1_2 = self.branch_1_2(branch_1)
        x1 = torch.cat([branch_1_1, branch_1_2], 1)

        branch_2 = self.branch_2(x)
        branch_2_1 = self.branch_2_1(branch_2)
        branch_2_2 = self.branch_2_2(branch_2)
        x2 = torch.cat([branch_2_1, branch_2_2], 1)

        x3 = self.branch_3(x)

        out = torch.cat([branch_0, x1, x2, x3], 1)

        return out
