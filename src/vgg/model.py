import torch
import torch.nn as nn
import torch.nn.functional as F


def calculate_padding(x):
    # Take last two dimensions (height / width)
    h, w = x.shape[-2:]

    pad_h = 0 if h % 2 == 0 else 1
    pad_w = 0 if w % 2 == 0 else 1

    return pad_h, pad_w


class PaddedConv2d(nn.Module):
    """
    Add zero-padding to nn.Conv2d to make it divisible by 2.
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1, stride=1):
        super(PaddedConv2d, self).__init__()

        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            padding=padding,
            stride=stride
        )

    def forward(self, x):
        pad_h, pad_w = calculate_padding(x)
        padded_x = F.pad(x, (0, pad_w, 0, pad_h))

        x = self.conv(padded_x)
        return x


class VGG(nn.Module):
    def __init__(self, in_channels, n_classes, layers):
        super(VGG, self).__init__()
        self.in_channels = in_channels

        self.layers = layers

        self.conv_blocks = self._build_conv_blocks()

        self.fc_blocks = nn.Sequential(
            nn.Flatten(),
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, n_classes)
        )

    def forward(self, x):
        x = self.conv_blocks(x)
        x = self.fc_blocks(x)
        return x

    def _build_conv_blocks(self):
        blocks = []

        in_channels = self.in_channels
        for layer in self.layers:
            if isinstance(layer, int):
                blocks += [
                    PaddedConv2d(in_channels=in_channels, out_channels=layer, kernel_size=3, padding=1),
                    nn.ReLU(inplace=True)
                ]

                in_channels = layer
            elif layer == 'M':
                blocks += [
                    nn.MaxPool2d(kernel_size=2, stride=2)
                ]

        return nn.Sequential(*blocks)


if __name__ == "__main__":
    x = torch.randn(4, 3, 224, 224)

    layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
    model = VGG(3, 1000, layers)

    out = model(x)

    print(out.shape)
