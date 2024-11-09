import torch
import torch.nn as nn



class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, *args, **kwargs):
        super().__init__(*args, **kwargs)

        
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)
    


class YOLOv1(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.conv1_1 = ConvBlock(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3)
        self.maxpool1_1 = nn.MaxPool2d(2, 2)

        self.conv1_2 = ConvBlock(in_channels=64, out_channels=192, kernel_size=3)  
        self.maxpool1_2 = nn.MaxPool2d(2, 2)

        self.conv2_1 = ConvBlock(in_channels=192, out_channels=128, kernel_size=1, padding=0)
        self.conv2_2 = ConvBlock(in_channels=128, out_channels=256, kernel_size=3)
        self.conv2_3 = ConvBlock(in_channels=256, out_channels=256, kernel_size=1, padding=0)
        self.conv2_4 = ConvBlock(in_channels=256, out_channels=512, kernel_size=3)
        self.maxpool2 = nn.MaxPool2d(2, 2)

        self.conv3 = nn.ModuleList([
            nn.Sequential(
                ConvBlock(512, 256, kernel_size=1, padding=0),
                ConvBlock(256, 512, kernel_size=3)
            )
            for _ in range(4)
        ])

        self.conv4_1 = ConvBlock(in_channels=512, out_channels=512, kernel_size=1, padding=0)
        self.conv4_2 = ConvBlock(in_channels=512, out_channels=1024, kernel_size=3)
        self.maxpool4 = nn.MaxPool2d(2, 2)

        self.conv5 = nn.ModuleList([
            nn.Sequential(
                ConvBlock(1024, 512, kernel_size=1, padding=0),
                ConvBlock(512, 1024, kernel_size=3)
            )
            for _ in range(2)
        ])

        self.conv6_1 = ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3)
        self.conv6_2 = ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=2)
        self.conv6_3 = ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3)
        self.conv6_4 = ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3)

        self.flatten = nn.Flatten()

        self.fc6_5 = nn.Sequential(
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(inplace=True)
        )
        self.dropout = nn.Dropout()
        self.fc6_6 = nn.Linear(4096, 7 * 7 * 30)


    def forward(self, x):
        x = self.conv1_1(x)
        x = self.maxpool1_1(x)
        x = self.conv1_2(x)
        x = self.maxpool1_2(x)

        x = self.conv2_1(x)
        x = self.conv2_2(x)
        x = self.conv2_3(x)
        x = self.conv2_4(x)
        x = self.maxpool2(x)

        for module in self.conv3:
            x = module(x)

        x = self.conv4_1(x)
        x = self.conv4_2(x)
        x = self.maxpool4(x)

        for module in self.conv5:
            x = module(x)

        x = self.conv6_1(x)
        x = self.conv6_2(x)
        x = self.conv6_3(x)
        x = self.conv6_4(x)

        x = self.flatten(x)

        x = self.fc6_5(x)
        x = self.dropout(x)
        x = self.fc6_6(x)

        x = x.reshape(-1, 7, 7, 30)
        return x
    


if __name__ == "__main__":
    x = torch.randn(4, 3, 448, 448)
    model = YOLOv1(in_channels=3)

    out = model(x)


