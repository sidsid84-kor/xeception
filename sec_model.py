from typing import Any
import torch
from torch import Tensor
from torch import nn
from torchvision.utils import save_image
import torch.nn.functional as F
from functools import partial
import math

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


if hasattr(nn, 'SiLU'):
    Swish = nn.SiLU
else:
    # For compatibility with old PyTorch versions
    class Swish(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)

class Sec_Effinet(nn.Module):

    def __init__(self, num_classes = 3, dropout_rate = 0.2, image_size = 640):
        super().__init__()
        
        import collections
        
        self.BlockArgs = collections.namedtuple('BlockArgs', [
            'num_repeat', 'kernel_size', 'stride', 'expand_ratio',
            'input_filters', 'output_filters', 'se_ratio', 'id_skip'])
        
        GlobalParams = collections.namedtuple('GlobalParams', [
            'width_coefficient', 'depth_coefficient', 'image_size', 'dropout_rate',
            'num_classes', 'batch_norm_momentum', 'batch_norm_epsilon',
            'drop_connect_rate', 'depth_divisor', 'min_depth', 'include_top'])
        
        self._global_params = GlobalParams(
            width_coefficient=2.2, depth_coefficient=1.0, image_size=image_size, 
            dropout_rate=dropout_rate, num_classes=num_classes, batch_norm_momentum=0.99, 
            batch_norm_epsilon=0.001, drop_connect_rate=0.2, depth_divisor=8, 
            min_depth=None, include_top=True
        )

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Get stem static or dynamic convolution depending on image size
        image_size = self._global_params.image_size
        Conv2d = get_same_padding_conv2d(image_size=image_size)

        # Stem
        in_channels = 1  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem_1 = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._conv_stem_2 = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0_1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._bn0_2 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        image_size = calculate_output_image_size(image_size, 2)

        self.MBblock_1_1 = MBConvBlock(
            self.BlockArgs(num_repeat=1, kernel_size=3, stride=[1], expand_ratio=1, input_filters=72, output_filters=32, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        self.MBblock_1_2 = MBConvBlock(
            self.BlockArgs(num_repeat=1, kernel_size=3, stride=[1], expand_ratio=1, input_filters=72, output_filters=32, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )

        self.MBblock_2_1 = MBConvBlock(
            self.BlockArgs(num_repeat=2, kernel_size=3, stride=[2], expand_ratio=6, input_filters=32, output_filters=56, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        image_size = [160, 160] # = calculate_output_image_size(image_size, 이전 stride)

        self.MBblock_2_2 = MBConvBlock(
            self.BlockArgs(num_repeat=2, kernel_size=3, stride=1, expand_ratio=6, input_filters=32, output_filters=56, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )

        self.MBblock_3_1 = MBConvBlock(
            self.BlockArgs(num_repeat=2, kernel_size=5, stride=[2], expand_ratio=6, input_filters=56, output_filters=88, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        image_size = [80, 80] # = calculate_output_image_size(image_size, 이전 stride [2])

        self.MBblock_3_2 = MBConvBlock(
            self.BlockArgs(num_repeat=2, kernel_size=5, stride=[2], expand_ratio=6, input_filters=56, output_filters=88, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        

        self.MBblock_4_1 = MBConvBlock(
            self.BlockArgs(num_repeat=3, kernel_size=3, stride=[2], expand_ratio=6, input_filters=88, output_filters=176, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )

        
        image_size = [40, 40] # = calculate_output_image_size(image_size, 이전 stride)

        self.MBblock_4_2 = MBConvBlock(
            self.BlockArgs(num_repeat=3, kernel_size=3, stride=[2], expand_ratio=6, input_filters=88, output_filters=176, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )
        
        self.MBblock_5_1 = MBConvBlock(
            self.BlockArgs(num_repeat=3, kernel_size=5, stride=[1], expand_ratio=6, input_filters=176, output_filters=248, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )

        self.MBblock_5_2 = MBConvBlock(
            self.BlockArgs(num_repeat=3, kernel_size=5, stride=[1], expand_ratio=6, input_filters=176, output_filters=248, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )

        self.MBblock_6_1 = MBConvBlock(
            self.BlockArgs(num_repeat=4, kernel_size=5, stride=[2], expand_ratio=6, input_filters=248, output_filters=424, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )


        image_size = [20, 20] # = calculate_output_image_size(image_size, 이전 stride)

        self.MBblock_6_2 = MBConvBlock(
            self.BlockArgs(num_repeat=4, kernel_size=5, stride=[2], expand_ratio=6, input_filters=248, output_filters=424, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )

        self.MBblock_7_1 = MBConvBlock(
            self.BlockArgs(num_repeat=1, kernel_size=3, stride=[1], expand_ratio=6, input_filters=424, output_filters=704, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )

        self.MBblock_7_2 = MBConvBlock(
            self.BlockArgs(num_repeat=1, kernel_size=3, stride=[1], expand_ratio=6, input_filters=424, output_filters=704, se_ratio=0.25, id_skip=True),
            self._global_params, image_size
        )

        # Head
        in_channels = 704 #block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._conv_head_1 = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1_1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)
        self._conv_head_2 = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1_2 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling_1 = nn.AdaptiveAvgPool2d(1)
        self._avg_pooling_2 = nn.AdaptiveAvgPool2d(1)
        
        if self._global_params.include_top:
            self._dropout = nn.Dropout(self._global_params.dropout_rate)
            # self._fc = nn.Linear(out_channels, self._global_params.num_classes)
            self._fc_1 = nn.Linear(out_channels*2, num_classes)
            

        # set activation to memory efficient swish by default
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export).

        Args:
            memory_efficient (bool): Whether to use memory-efficient version of swish.
        """
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_endpoints(self, inputs):
        """Use convolution layer to extract features
        from reduction levels i in [1, 2, 3, 4, 5].

        Args:
            inputs (tensor): Input tensor.

        Returns:
            Dictionary of last intermediate features
            with reduction levels i in [1, 2, 3, 4, 5].
            Example:
                >>> import torch
                >>> from efficientnet.model import EfficientNet
                >>> inputs = torch.rand(1, 3, 224, 224)
                >>> model = EfficientNet.from_pretrained('efficientnet-b0')
                >>> endpoints = model.extract_endpoints(inputs)
                >>> print(endpoints['reduction_1'].shape)  # torch.Size([1, 16, 112, 112])
                >>> print(endpoints['reduction_2'].shape)  # torch.Size([1, 24, 56, 56])
                >>> print(endpoints['reduction_3'].shape)  # torch.Size([1, 40, 28, 28])
                >>> print(endpoints['reduction_4'].shape)  # torch.Size([1, 112, 14, 14])
                >>> print(endpoints['reduction_5'].shape)  # torch.Size([1, 320, 7, 7])
                >>> print(endpoints['reduction_6'].shape)  # torch.Size([1, 1280, 7, 7])
        """
        endpoints = dict()

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))
        prev_x = x

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)  # scale drop connect_rate
            x = block(x, drop_connect_rate=drop_connect_rate)
            if prev_x.size(2) > x.size(2):
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = prev_x
            elif idx == len(self._blocks) - 1:
                endpoints['reduction_{}'.format(len(endpoints) + 1)] = x
            prev_x = x

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))
        endpoints['reduction_{}'.format(len(endpoints) + 1)] = x

        return endpoints


    def forward(self, inputs):
        
        x1 = inputs[:, 1, :, :].unsqueeze(1)  # [2, 640, 640] → [2, 1, 640, 640]
        x2 = inputs[:, 2, :, :].unsqueeze(1)  # [2, 640, 640] → [2, 1, 640, 640]
        
        x1 = self._swish(self._bn0_1(self._conv_stem_1(x1)))
        x2 = self._swish(self._bn0_2(self._conv_stem_2(x2)))
        
        # Blocks
        x1 = self.MBblock_1_1(x1)
        x2 = self.MBblock_1_2(x2)

        x1 = self.MBblock_2_1(x1)
        x2 = self.MBblock_2_2(x2)

        x1 = self.MBblock_3_1(x1)
        x2 = self.MBblock_3_2(x2)

        x1 = self.MBblock_4_1(x1)
        x2 = self.MBblock_4_2(x2)

        x1 = self.MBblock_5_1(x1)
        x2 = self.MBblock_5_2(x2)

        x1 = self.MBblock_6_1(x1)
        x2 = self.MBblock_6_2(x2)

        x1 = self.MBblock_7_1(x1)
        x2 = self.MBblock_7_1(x2)

        # Head
        x1 = self._swish(self._bn1_1(self._conv_head_1(x1)))
        x2 = self._swish(self._bn1_1(self._conv_head_2(x2)))
        # Pooling and final linear layer

        x1 = self._avg_pooling_1(x1)
        x2 = self._avg_pooling_2(x2)
        
        x1 = torch.flatten(x1, 1)
        x2 = torch.flatten(x2, 1)
        
        combined = torch.cat([x1, x2], dim=1)  # 채널 방향으로 결합
        

        # Flatten before passing to fully connected layers
        combined = combined.view(combined.size(0), -1)

               

        # Fully connected layer for classification
        final_outputs = self._fc_1(combined)
        

        return final_outputs



class MBConvBlock(nn.Module):
    def __init__(self, block_args, global_params, image_size=None):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum  # pytorch's difference from tensorflow
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # whether to use skip connection and drop connect

        # Expansion phase (Inverted Bottleneck)
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            Conv2d = get_same_padding_conv2d(image_size=image_size)
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
            # image_size = calculate_output_image_size(image_size, 1) <-- this wouldn't modify image_size

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)
        image_size = calculate_output_image_size(image_size, s)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            Conv2d = get_same_padding_conv2d(image_size=(1, 1))
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Pointwise convolution phase
        final_oup = self._block_args.output_filters
        Conv2d = get_same_padding_conv2d(image_size=image_size)
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._expand_conv(inputs)
            x = self._bn0(x)
            x = self._swish(x)

        x = self._depthwise_conv(x)
        x = self._bn1(x)
        x = self._swish(x)

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_reduce(x_squeezed)
            x_squeezed = self._swish(x_squeezed)
            x_squeezed = self._se_expand(x_squeezed)
            x = torch.sigmoid(x_squeezed) * x

        # Pointwise Convolution
        x = self._project_conv(x)
        x = self._bn2(x)

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            # The combination of skip connection and drop connect brings about stochastic depth.
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


def get_same_padding_conv2d(image_size=None):
    """Chooses static padding if you have specified an image size, and dynamic padding otherwise.
       Static padding is necessary for ONNX exporting of models.

    Args:
        image_size (int or tuple): Size of the image.

    Returns:
        Conv2dDynamicSamePadding or Conv2dStaticSamePadding.
    """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)
        
    
class Conv2dStaticSamePadding(nn.Conv2d):
    """2D Convolutions like TensorFlow's 'SAME' mode, with the given input image size.
       The padding mudule is calculated in construction function, then used in forward.
    """

    # With the same calculation as Conv2dDynamicSamePadding

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, stride, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = (image_size, image_size) if isinstance(image_size, int) else image_size
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2,
                                                pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = nn.Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x
    
class Conv2dDynamicSamePadding(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)  # change the output size according to stride ! ! !
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
    
def round_filters(filters, global_params):
    """Calculate and round number of filters based on width multiplier.
       Use width_coefficient, depth_divisor and min_depth of global_params.

    Args:
        filters (int): Filters number to be calculated.
        global_params (namedtuple): Global params of the model.

    Returns:
        new_filters: New filters number after calculating.
    """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    # TODO: modify the params names.
    #       maybe the names (width_divisor,min_width)
    #       are more suitable than (depth_divisor,min_depth).
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor  # pay attention to this line when using min_depth
    # follow the formula transferred from official TensorFlow implementation
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)

def calculate_output_image_size(input_image_size, stride):
    """Calculates the output image size when using Conv2dSamePadding with a stride.
       Necessary for static padding. Thanks to mannatsingh for pointing this out.

    Args:
        input_image_size (int, tuple or list): Size of input image.
        stride (int, tuple or list): Conv2d operation's stride.

    Returns:
        output_image_size: A list [H,W].
    """
    if input_image_size is None:
        return None
    image_height, image_width = get_width_and_height_from_size(input_image_size)
    stride = stride if isinstance(stride, int) else stride[0]
    image_height = int(math.ceil(image_height / stride))
    image_width = int(math.ceil(image_width / stride))
    return [image_height, image_width]


def get_width_and_height_from_size(x):
    """Obtain height and width from x.

    Args:
        x (int, tuple or list): Data size.

    Returns:
        size: A tuple or list (H,W).
    """
    if isinstance(x, int):
        return x, x
    if isinstance(x, list) or isinstance(x, tuple):
        return x
    else:
        raise TypeError()
    
class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)
    
class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_tensors[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))