from typing import Any
import torch
from torch import Tensor
from torch import nn
from torchvision.utils import save_image

class InceptionV4_parallel(nn.Module):

    def __init__(
            self,
            k: int = 96,
            l: int = 112,
            m: int = 128,
            n: int = 192,
            num_classes: int = 1000,
            dropout_prob = 0.0,
    ) -> None:
        super(InceptionV4_parallel, self).__init__()
        
        # Stem blocks (2개 채널용)
        self.stem1 = InceptionV4Stem(1)
        self.stem2 = InceptionV4Stem(1)
        
        # Inception-A blocks (2개 채널용)
        self.inception_a1_1 = InceptionA(192)
        self.inception_a1_2 = InceptionA(192)
        
        self.inception_a2_1 = InceptionA(192)
        self.inception_a2_2 = InceptionA(192)
        
        self.inception_a3_1 = InceptionA(192)
        self.inception_a3_2 = InceptionA(192)
        
        self.inception_a4_1 = InceptionA(192)
        self.inception_a4_2 = InceptionA(192)
        
        # Reduction-A blocks
        self.reduction_a1 = ReductionA(192, k, l, m, n)
        self.reduction_a2 = ReductionA(192, k, l, m, n)
        
        # Inception-B blocks
        self.inception_b1_1 = InceptionB(512)
        self.inception_b1_2 = InceptionB(512)

        self.inception_b2_1 = InceptionB(512)
        self.inception_b2_2 = InceptionB(512)

        self.inception_b3_1 = InceptionB(512)
        self.inception_b3_2 = InceptionB(512)
        
        self.inception_b4_1 = InceptionB(512)
        self.inception_b4_2 = InceptionB(512)

        self.inception_b5_1 = InceptionB(512)
        self.inception_b5_2 = InceptionB(512)

        self.inception_b6_1 = InceptionB(512)
        self.inception_b6_2 = InceptionB(512)

        self.inception_b7_1 = InceptionB(512)
        self.inception_b7_2 = InceptionB(512)
        
        # Reduction-B blocks
        self.reduction_b1 = ReductionB(512)
        self.reduction_b2 = ReductionB(512)
        
        # Inception-C blocks
        self.inception_c1_1 = InceptionC(768)
        self.inception_c1_2 = InceptionC(768)

        self.inception_c2_1 = InceptionC(768)
        self.inception_c2_2 = InceptionC(768)

        self.inception_c3_1 = InceptionC(768)
        self.inception_c3_2 = InceptionC(768)
        

        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout = nn.Dropout(dropout_prob)
        
        self.global_average_pooling2 = nn.AdaptiveAvgPool2d((1, 1))

        self.dropout2 = nn.Dropout(dropout_prob)

        self.linear = nn.Linear(768 * 2, num_classes)

        # Initialize neural network weights
        self._initialize_weights()

    def forward(self, x: Tensor) -> Tensor:
        
        x1 = x[:, 1, :, :].unsqueeze(1)  # [2, 640, 640] → [2, 1, 640, 640]
        x2 = x[:, 2, :, :].unsqueeze(1)  # [2, 640, 640] → [2, 1, 640, 640]

        # 채널 1 처리
        out1 = self.stem1(x1)
        out1 = self.inception_a1_1(out1)
        out1 = self.inception_a2_1(out1)
        out1 = self.inception_a3_1(out1)
        out1 = self.inception_a4_1(out1)
        out1 = self.reduction_a1(out1)
        out1 = self.inception_b1_1(out1)
        out1 = self.inception_b2_1(out1)
        out1 = self.inception_b3_1(out1)
        out1 = self.inception_b4_1(out1)
        out1 = self.reduction_b1(out1)
        out1 = self.inception_c1_1(out1)
        out1 = self.inception_c2_1(out1)
        out1 = self.inception_c3_1(out1)

        # 채널 2 처리
        out2 = self.stem2(x2)
        out2 = self.inception_a1_2(out2)
        out2 = self.inception_a2_2(out2)
        out2 = self.inception_a3_2(out2)
        out2 = self.inception_a4_2(out2)
        out2 = self.reduction_a2(out2)
        out2 = self.inception_b1_2(out2)
        out2 = self.inception_b2_2(out2)
        out2 = self.inception_b3_2(out2)
        out2 = self.inception_b4_2(out2)
        out2 = self.reduction_b2(out2)
        out2 = self.inception_c1_2(out2)
        out2 = self.inception_c2_2(out2)
        out2 = self.inception_c3_2(out2)

        # 병합
        out1 = self.global_average_pooling(out1)
        out2 = self.global_average_pooling2(out2)
        
        out1 = self.dropout(out1)
        out2 = self.dropout2(out2)
        
        combined = torch.cat([out1, out2], dim=1)  # 채널 방향으로 결합
        

        # Flatten before passing to fully connected layers
        combined = combined.view(combined.size(0), -1)

               

        # Fully connected layer for classification
        out = self.linear(combined)
        return out

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
        self.conv2d_1a_3x3 = BasicConv2d(in_channels, 16, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))

        self.conv2d_2a_3x3 = BasicConv2d(16, 16, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        self.conv2d_2b_3x3 = BasicConv2d(16, 32, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

        self.mixed_3a_branch_0 = nn.MaxPool2d((3, 3), (2, 2))
        self.mixed_3a_branch_1 = BasicConv2d(32, 48, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))

        self.mixed_4a_branch_0 = nn.Sequential(
            BasicConv2d(80, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(32, 48, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0)),
        )
        self.mixed_4a_branch_1 = nn.Sequential(
            BasicConv2d(80, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(32, 32, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
            BasicConv2d(32, 32, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
            BasicConv2d(32, 48, kernel_size=(3, 3), stride=(1, 1), padding=(0, 0))
        )

        self.mixed_5a_branch_0 = BasicConv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0))
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


class InceptionA(nn.Module):
    def __init__(
            self,
            in_channels: int,
    ) -> None:
        super(InceptionA, self).__init__()
        self.branch_0 = BasicConv2d(in_channels, 48, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(32, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.branch_2 = nn.Sequential(
            BasicConv2d(in_channels, 32, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(32, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            BasicConv2d(48, 48, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d((3, 3), (1, 1), (1, 1), count_include_pad=False),
            BasicConv2d(in_channels, 48, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        )

    def forward(self, x: Tensor) -> Tensor:
        branch_0 = self.branch_0(x)
        branch_1 = self.branch_1(x)
        branch_2 = self.branch_2(x)
        branch_3 = self.branch_3(x)

        out = torch.cat([branch_0, branch_1, branch_2, branch_3], 1)

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
        self.branch_0 = BasicConv2d(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, 96, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(96, 112, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
            BasicConv2d(112, 128, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
        )
        self.branch_2 = nn.Sequential(
            BasicConv2d(in_channels, 96, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(96, 96, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
            BasicConv2d(96, 112, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
            BasicConv2d(112, 112, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
            BasicConv2d(112, 128, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
        )
        self.branch_3 = nn.Sequential(
            nn.AvgPool2d((3, 3), (1, 1), (1, 1), count_include_pad=False),
            BasicConv2d(in_channels, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
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
            BasicConv2d(in_channels, 96, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(96, 96, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
        )
        self.branch_1 = nn.Sequential(
            BasicConv2d(in_channels, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(128, 128, kernel_size=(1, 7), stride=(1, 1), padding=(0, 3)),
            BasicConv2d(128, 160, kernel_size=(7, 1), stride=(1, 1), padding=(3, 0)),
            BasicConv2d(160, 160, kernel_size=(3, 3), stride=(2, 2), padding=(0, 0)),
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
        self.branch_0 = BasicConv2d(in_channels, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))

        self.branch_1 = BasicConv2d(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        self.branch_1_1 = BasicConv2d(192, 128, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.branch_1_2 = BasicConv2d(192, 128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))

        self.branch_2 = nn.Sequential(
            BasicConv2d(in_channels, 192, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            BasicConv2d(192, 224, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0)),
            BasicConv2d(224, 256, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1)),
        )
        self.branch_2_1 = BasicConv2d(256, 128, kernel_size=(1, 3), stride=(1, 1), padding=(0, 1))
        self.branch_2_2 = BasicConv2d(256, 128, kernel_size=(3, 1), stride=(1, 1), padding=(1, 0))

        self.branch_3 = nn.Sequential(
            nn.AvgPool2d((3, 3), (1, 1), (1, 1)),
            BasicConv2d(in_channels, 128, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
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

