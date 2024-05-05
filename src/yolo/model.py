import torch
import torch.nn as nn


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=0, stride=1):
        super(ConvBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding, stride=stride),
            nn.LeakyReLU(negative_slope=0.1, inplace=True)
        )

    def forward(self, x):
        return self.block(x)


class YOLO(nn.Module):
    def __init__(self, in_channels):
        super(YOLO, self).__init__()

        self.block1 = nn.Sequential(
            ConvBlock(in_channels=in_channels, out_channels=64, kernel_size=7, stride=2, padding=3),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block2 = nn.Sequential(
            ConvBlock(in_channels=64, out_channels=192, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block3 = nn.Sequential(
            ConvBlock(in_channels=192, out_channels=128, kernel_size=1),
            ConvBlock(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            ConvBlock(in_channels=256, out_channels=256, kernel_size=1),
            ConvBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block4 = nn.Sequential(
            *[
                nn.Sequential(
                    ConvBlock(in_channels=512, out_channels=256, kernel_size=1),
                    ConvBlock(in_channels=256, out_channels=512, kernel_size=3, padding=1),
                )
                for _ in range(4)
            ],
            ConvBlock(in_channels=512, out_channels=512, kernel_size=1),
            ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.block5 = nn.Sequential(
            *[
                nn.Sequential(
                    ConvBlock(in_channels=1024, out_channels=512, kernel_size=1),
                    ConvBlock(in_channels=512, out_channels=1024, kernel_size=3, padding=1),
                )
                for _ in range(2)
            ],
            ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, stride=2, padding=1),
        )

        self.block6 = nn.Sequential(
            ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
            ConvBlock(in_channels=1024, out_channels=1024, kernel_size=3, padding=1),
        )

        self.fc_block = nn.Sequential(
            nn.Flatten(),
            nn.Linear(7 * 7 * 1024, 4096),
            nn.LeakyReLU(negative_slope=0.1, inplace=True),
            nn.Linear(4096, 7 * 7 * 30)
        )

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)

        x = self.fc_block(x)
        return x


if __name__ == '__main__':
    model = YOLO(in_channels=3)

    data = torch.randn(1, 3, 448, 448)
    result = model(data)

    print(result.shape)