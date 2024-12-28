import torch
import torch.nn as nn
from config import CONFIG


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
    def __init__(self, in_channels, n_classes, alpha=1.0):
        super().__init__()

        self._config = CONFIG
        self.in_channels = in_channels
        self.n_classes = n_classes
        self.alpha = alpha

        self.model = self._build_model(self._config)

    def _build_model(self, config):
        layers = []
        in_channels = self.in_channels

        for layer in config['layers']:
            if layer['type'] == 'conv':
                layers.append(ConvBlock(in_channels, **layer['params']))
                in_channels = layer['params']['out_channels']
            elif layer['type'] == 'dw_conv':
                layers.append(DepthWiseConv2d(in_channels, **layer['params']))
                in_channels = layer['params']['out_channels']
            elif layer['type'] == 'flatten':
                layers.append(nn.Flatten(**layer['params']))
            elif layer['type'] == 'avg_pool':
                layers.append(nn.AvgPool2d(**layer['params']))
            elif layer['type'] == 'fc':
                layers.append(nn.Linear(out_features=self.n_classes, **layer['params']))

        return nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


if __name__ == "__main__":
    x = torch.rand((1, 3, 224, 224))

    model = MobileNet(3, 5)
    out = model(x)
    print(out.shape)
