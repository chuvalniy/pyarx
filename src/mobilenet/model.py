import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride=1,
            padding=1,
            groups=1,
    ):
        super().__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups=groups),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class DepthWiseConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super().__init__()

        self.dw_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_channels
        )
        self.pw_conv = ConvBlock(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )

    def forward(self, x):
        x = self.dw_conv(x)
        x = self.pw_conv(x)
        return x


class MobileNet(nn.Module):
    def __init__(self, in_channels, out_classes):
        super().__init__()

        self.conv_1 = ConvBlock(in_channels, 32, 3, 2, 1)
        self.dw_conv_1 = DepthWiseConv2d(32, 32, 3, 1, 1)

        self.conv_2 = ConvBlock(32, 64, 1, 1, 0)
        self.dw_conv_2 = DepthWiseConv2d(64, 64, 3, 2, 1)

        self.conv_3_1 = ConvBlock(64, 128, 1, 1, 0)
        self.dw_conv_3_1 = DepthWiseConv2d(128, 128, 3, 1, 1)
        self.conv_3_2 = ConvBlock(128, 128, 1, 1, 0)
        self.dw_conv_3_2 = DepthWiseConv2d(128, 128, 3, 2, 1)

        self.conv_4_1 = ConvBlock(128, 256, 1, 1, 0)
        self.dw_conv_4_1 = DepthWiseConv2d(256, 256, 3, 1, 1)
        self.conv_4_2 = ConvBlock(256, 256, 1, 1, 0)
        self.dw_conv_4_2 = DepthWiseConv2d(256, 256, 3, 2, 1)
        self.conv_4_3 = ConvBlock(256, 512, 1, 1, 0)

        self.conv5 = nn.ModuleList([
            nn.Sequential(
                DepthWiseConv2d(512, 512, 3, 1, 1),
                ConvBlock(512, 512, 1, 1, 0)
            ) for _ in range(5)
        ])

        self.dw_conv_6_1 = DepthWiseConv2d(512, 512, 3, 2, 1)
        self.conv_6_1 = ConvBlock(512, 1024, 1, 1, 0)
        self.dw_conv_6_2 = DepthWiseConv2d(1024, 1024, 3, 2, 4)
        self.conv_6_2 = ConvBlock(1024, 1024, 1, 1, 0)

        self.avg_pool = nn.AvgPool2d(kernel_size=7, stride=1)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(1024, out_classes)

    def forward(self, x):
        x = self.conv_1(x)
        x = self.dw_conv_1(x)
        x = self.conv_2(x)
        x = self.dw_conv_2(x)
        x = self.conv_3_1(x)
        x = self.dw_conv_3_1(x)
        x = self.conv_3_2(x)
        x = self.dw_conv_3_2(x)
        x = self.conv_4_1(x)
        x = self.dw_conv_4_1(x)
        x = self.conv_4_2(x)
        x = self.dw_conv_4_2(x)
        x = self.conv_4_3(x)

        for conv in self.conv5:
            x = conv(x)

        x = self.dw_conv_6_1(x)
        x = self.conv_6_1(x)
        x = self.dw_conv_6_2(x)
        x = self.conv_6_2(x)
        x = self.avg_pool(x)
        x = self.flatten(x)
        x = self.fc(x)

        return x


if __name__ == "__main__":
    x = torch.rand((1, 3, 224, 224))

    model = MobileNet(3, 5)
    out = model(x)
