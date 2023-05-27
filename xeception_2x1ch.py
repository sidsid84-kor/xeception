import torch
import torch.nn as nn

class ChannelSplit(nn.Module):
    def __init__(self):
        super(ChannelSplit, self).__init__()

    def forward(self, x):
        x1, x2, x3 = torch.chunk(x, chunks=3, dim=1)
        return x1, x2, x3

class SeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.seperable = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, stride=1, padding=1, groups=in_channels, bias=False),
            nn.Conv2d(in_channels, out_channels, 1, stride=1, padding=0, bias=False)
        )

    def forward(self, x):
        x = self.seperable(x)
        return x
    
# EnrtyFlow
class EntryFlow(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 16, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 32, 3, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU()
        )

        self.conv2_residual = nn.Sequential(
            SeparableConv(32, 64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            SeparableConv(64, 64),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.conv2_shortcut = nn.Sequential(
            nn.Conv2d(32, 64, 1, stride=2, padding=0),
            nn.BatchNorm2d(64)
        )

        self.conv3_residual = nn.Sequential(
            nn.ReLU(),
            SeparableConv(64, 128),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            SeparableConv(128, 128),
            nn.BatchNorm2d(128),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.conv3_shortcut = nn.Sequential(
            nn.Conv2d(64, 128, 1, stride=2, padding=0),
            nn.BatchNorm2d(128)
        )

        self.conv4_residual = nn.Sequential(
            nn.ReLU(),
            SeparableConv(128, 364),
            nn.BatchNorm2d(364),
            nn.ReLU(),
            SeparableConv(364, 364),
            nn.BatchNorm2d(364),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.conv4_shortcut = nn.Sequential(
            nn.Conv2d(128, 364, 1, stride=2, padding=0),
            nn.BatchNorm2d(364)
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2_residual(x) + self.conv2_shortcut(x)
        x = self.conv3_residual(x) + self.conv3_shortcut(x)
        x = self.conv4_residual(x) + self.conv4_shortcut(x)
        return x
    
# MiddleFlow
class MiddleFlow(nn.Module):
    def __init__(self):
        super().__init__()

        self.conv_residual = nn.Sequential(
            nn.ReLU(),
            SeparableConv(364, 364),
            nn.BatchNorm2d(364),
            nn.ReLU(),
            SeparableConv(364, 364),
            nn.BatchNorm2d(364),
            nn.ReLU(),
            SeparableConv(364, 364),
            nn.BatchNorm2d(364)
        )

        self.conv_shortcut = nn.Sequential()

    def forward(self, x):
        return self.conv_shortcut(x) + self.conv_residual(x)
    
# ExitFlow
class ExitFlow(nn.Module):
    def __init__(self, num_classes=10):
        super().__init__()

        self.conv1_residual = nn.Sequential(
            nn.ReLU(),
            SeparableConv(364, 512),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            SeparableConv(512, 512),
            nn.BatchNorm2d(512),
            nn.MaxPool2d(3, stride=2, padding=1)
        )

        self.conv1_shortcut = nn.Sequential(
            nn.Conv2d(364, 512, 1, stride=2, padding=0),
            nn.BatchNorm2d(512)
        )

        self.conv2 = nn.Sequential(
            SeparableConv(512, 768),
            nn.BatchNorm2d(768),
            nn.ReLU(),
            SeparableConv(768, 1024),
            nn.BatchNorm2d(1024),
            nn.ReLU()
        )

        self.avg_pool = nn.AdaptiveAvgPool2d((1,1))
    
    def forward(self, x):
        x = self.conv1_residual(x) + self.conv1_shortcut(x)
        x = self.conv2(x)
        x = self.avg_pool(x)
        return x
# Xception
class Xception2x1ch(nn.Module):
    def __init__(self, num_classes=10, init_weights=True):
        super().__init__()
        self.init_weights = init_weights

        self.channel_split = ChannelSplit()

        self.entry = EntryFlow()
        self.middle = self._make_middle_flow()
        self.exit = ExitFlow()

        self.linear = nn.Linear(1024, 128)

        self.confc = nn.Linear(256, num_classes)

        # weights initialization
        if self.init_weights:
            pass


    def forward(self, x):
        _, x1, x2 = self.channel_split(x)

        x1 = self.entry(x1)
        x2 = self.entry(x2)

        x1 = self.middle(x1)
        x2 = self.middle(x2)

        x1 = self.exit(x1)
        x2 = self.exit(x2)

        x1 = x1.view(x1.size(0), -1)
        x2 = x1.view(x2.size(0), -1)

        x1 = self.linear(x1)
        x2 = self.linear(x2)
        
        out = torch.cat((x1, x2), dim=1)
        out = self.confc(out)
        
        return out





    def _make_middle_flow(self):
        middle = nn.Sequential()
        for i in range(8):
            middle.add_module('middle_block_{}'.format(i), MiddleFlow())
        return middle

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init_kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init_constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init_constant_(m.weight, 1)
                nn.init_bias_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init_normal_(m.weight, 0, 0.01)
                nn.init_constant_(m.bias, 0)
