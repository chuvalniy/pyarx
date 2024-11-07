import torch
import torch.nn as nn
from postprocessor import YOLOv2PostProcessor


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: tuple[int, int], stride: int, padding: int, *args, **kwargs):
        super().__init__(*args, **kwargs)


        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding),
            nn.LeakyReLU(inplace=True),
            nn.BatchNorm2d(out_channels)   
        )

    def forward(self, x):
        x = self.block(x)
        return x
  


class YOLOv2(nn.Module):
    def __init__(self, in_channels: int, n_classes, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv1 = ConvBlock(3, 32, (3, 3), 1, 1)
        self.pool1 = nn.MaxPool2d(2, 2)

        self.conv2 = ConvBlock(32, 64, (3, 3), 1, 1)
        self.pool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.Sequential(
            ConvBlock(64, 128, (3, 3), 1, 1),
            ConvBlock(128, 64, (1, 1), 1, 0),
            ConvBlock(64, 128, (3, 3), 1, 1)
        )
        self.pool3 = nn.MaxPool2d(2, 2)

        self.conv4 = nn.Sequential(
            ConvBlock(128, 256, (3, 3), 1, 1),
            ConvBlock(256, 128, (1, 1), 1, 0),
            ConvBlock(128, 256, (3, 3), 1, 1)
        )
        self.pool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.Sequential(
            ConvBlock(256, 512, (3, 3), 1, 1),
            ConvBlock(512, 256, (1, 1), 1, 0),
            ConvBlock(256, 512, (3, 3), 1, 1),
            ConvBlock(512, 256, (1, 1), 1, 0),
            ConvBlock(256, 512, (3, 3), 1, 1)
        )
        self.pool5 = nn.MaxPool2d(2, 2)

        self.conv6 = nn.Sequential(
            ConvBlock(512, 1024, (3, 3), 1, 1),
            ConvBlock(1024, 512, (1, 1), 1, 0),
            ConvBlock(512, 1024, (3, 3), 1, 1),
            ConvBlock(1024, 512, (1, 1), 1, 0),
            ConvBlock(512, 1024, (3, 3), 1, 1)
        )

        self.conv7 = nn.Sequential(
            ConvBlock(1024, 1024, (3, 3), 1, 1),
            ConvBlock(1024, 1024, (3, 3), 1, 1),
        )
        self.conv8 = ConvBlock(3072, 1024, (3, 3), 1, 1)
        self.conv9 = nn.Conv2d(1024, (5 + n_classes) * 5, (1, 1), 1, 0)


    def forward(self, x):
        # Backbone
        x = self.pool1(self.conv1(x))
        x = self.pool2(self.conv2(x))
        x = self.pool3(self.conv3(x))
        x = self.pool4(self.conv4(x))
        x = self.conv5(x)
        x_copy = x.clone()
        x = self.pool5(x)
        x = self.conv6(x)

        # Head
        x = self.conv7(x)
        x_copy = x_copy.reshape(-1, 2048, 13, 13)
        x = torch.concat((x_copy, x), dim=1)
        x = self.conv8(x)
        x = self.conv9(x)

        return x

    

# Loss
# Clustering of anchor boxes

if __name__ == "__main__":
    x = torch.randn(4, 3, 416, 416)

    model = YOLOv2(3, 20)
    out = model(x)
        
    postprocessor = YOLOv2PostProcessor()
    out = postprocessor.forward(out)

    print(out.shape)