import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.BatchNorm2d(out_channels),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    

class ResidualBlock(nn.Module):
    def __init__(self, n_channels, kernel_size, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = ConvBlock(n_channels, n_channels, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(n_channels, n_channels, kernel_size=kernel_size)

    def forward(self, x):
        out = self.conv1(x)
        out = self.conv2(out)

        return out + x
    

class DarkNet53(nn.Module):
    def __init__(self, in_channels: int = 3, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=32, kernel_size=3),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3, stride=2)
        )

        self.res_block1 = nn.Sequential(
            ConvBlock(in_channels=64, out_channels=32, kernel_size=1, padding=0),
            ConvBlock(in_channels=32, out_channels=64, kernel_size=3),
            ResidualBlock(n_channels=64, kernel_size=3),
        )
        self.cbl1 = ConvBlock(in_channels=64, out_channels=128, kernel_size=3, stride=2)

        self.res_block2 = nn.ModuleList([
            nn.Sequential(
                ConvBlock(in_channels=128, out_channels=64, kernel_size=1, padding=0),
                ConvBlock(in_channels=64, out_channels=128, kernel_size=3),
                ResidualBlock(n_channels=128, kernel_size=3),
            ) for _ in range(2)
        ])
        self.cbl2 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3, stride=2)

        self.res_block3 = nn.ModuleList([
            nn.Sequential(
                ConvBlock(in_channels=256, out_channels=128, kernel_size=1, padding=0),
                ConvBlock(in_channels=128, out_channels=256, kernel_size=3),
                ResidualBlock(n_channels=256, kernel_size=3),
            ) for _ in range(8)
        ])
        self.cbl3 = ConvBlock(in_channels=256, out_channels=512, kernel_size=3, stride=2)

        self.res_block4 = nn.ModuleList([
            nn.Sequential(
                ConvBlock(in_channels=512, out_channels=256, kernel_size=1, padding=0),
                ConvBlock(in_channels=256, out_channels=512, kernel_size=3),
                ResidualBlock(n_channels=512, kernel_size=3),
            ) for _ in range(8)
        ])
        self.cbl4 = ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, stride=2)

        self.res_block5 = nn.ModuleList([
            nn.Sequential(
                ConvBlock(in_channels=1024, out_channels=512, kernel_size=1, padding=0),
                ConvBlock(in_channels=512, out_channels=1024, kernel_size=3),
                ResidualBlock(n_channels=1024, kernel_size=3),
            ) for _ in range(4)
        ])


    def forward(self, x):
        x = self.conv(x)

        for block in self.res_block1:
            x = block(x)
        x = self.cbl1(x)

        for block in self.res_block2:
            x = block(x)
        x = self.cbl2(x)

        x1 = x.clone()
        for block in self.res_block3:
            x1 = block(x1)
        x = self.cbl3(x1)

        x2 = x.clone()
        for block in self.res_block4:
            x2 = block(x2)
        x = self.cbl4(x2)

        x3 = x.clone()
        for block in self.res_block5:
            x3 = block(x3)

        return x1, x2, x3

class SPPBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k=(5, 9, 13), *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = ConvBlock(in_channels=in_channels, out_channels=out_channels, kernel_size=1, padding=0)
        self.conv2 = ConvBlock(in_channels=out_channels * 4, out_channels=out_channels, kernel_size=1, padding=0)

        self.pool1 = nn.MaxPool2d(kernel_size=k[0], stride=1, padding=k[0] // 2)
        self.pool2 = nn.MaxPool2d(kernel_size=k[1], stride=1, padding=k[1] // 2)
        self.pool3 = nn.MaxPool2d(kernel_size=k[2], stride=1, padding=k[2] // 2)


    def forward(self, x):
        x = self.conv1(x)

        x1 = self.pool1(x)
        x2 = self.pool2(x)
        x3 = self.pool3(x)

        x = torch.cat([x, x1, x2, x3], dim=1)
        x = self.conv2(x)
        return x3


class UpsampleBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, scale: int = 2, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = ConvBlock(in_channels, out_channels, kernel_size=1, padding=0)
        self.upsample = nn.Upsample(scale_factor=scale)

    def forward(self, x):
        return self.upsample(self.conv(x))


class YOLOv3Head(nn.Module):
    def __init__(self, n_classes: int, n_anchors: int = 3, *args, **kwargs):
        super().__init__(*args , **kwargs)
        
        self.conv1 = ConvBlock(512, 512, kernel_size=1, padding=0)
        self.out1 = nn.Sequential(
            ConvBlock(512, 1024, kernel_size=3),
            nn.Conv2d(1024, n_anchors * (n_classes + 5), kernel_size=1, padding=0)
        )
        self.upsample1 = UpsampleBlock(512, 512)

        self.conv2 = ConvBlock(512, 512, kernel_size=1, padding=0)
        self.out2 = nn.Sequential(
            ConvBlock(1024, 2048, kernel_size=3),
            nn.Conv2d(2048, n_anchors * (n_classes + 5), kernel_size=1, padding=0)
        )
        self.upsample2 = UpsampleBlock(1024, 512)

        self.conv3 = ConvBlock(256, 512, kernel_size=1, padding=0)
        self.out3 = nn.Sequential(
            ConvBlock(1024, 2048, kernel_size=3),
            nn.Conv2d(2048, n_anchors * (n_classes + 5), kernel_size=1, padding=0)
        )



    def forward(self, x1, x2, x3):
        x3 = self.conv1(x3)
        x3_out = self.out1(x3)
        x3 = self.upsample1(x3)

        x2 = self.conv2(x2)
        x2 = torch.concat([x2, x3], dim=1)
        x2_out = self.out2(x2)
        x2 = self.upsample2(x2)

        x1 = self.conv3(x1)
        x1 = torch.concat([x2, x1], dim=1)
        x1_out = self.out3(x1)

        return x1_out, x2_out, x3_out


class YOLOv3(nn.Module):
    def __init__(self, in_channels: int = 3, n_classes: int = 80, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = DarkNet53(in_channels)
        self.spp = SPPBlock(1024, 512)
        self.head = YOLOv3Head(n_classes=n_classes)


    def forward(self, x):
        x1, x2, x3 = self.backbone(x)
        x3 = self.spp(x3)
        x1, x2, x3 = self.head(x1, x2, x3)
        return x1, x2, x3


if __name__ == '__main__':
    x = torch.rand(4, 3, 416, 416)

    model = YOLOv3(3, 80)
    outs = model(x)