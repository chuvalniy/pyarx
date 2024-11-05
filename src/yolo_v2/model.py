import torch
import torch.nn as nn

class DarkNetConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, n_bottleneck: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.conv = ConvBlock(in_channels, out_channels, 3, stride=1, padding=1)
        self.bottleneck = nn.ModuleList([Bottleneck(out_channels, in_channels) for _ in range(n_bottleneck)])


    def forward(self, x):
        out = self.conv(x)
        for module in self.bottleneck:
            out = module(out)

        return out


class Bottleneck(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.bottleneck = nn.Sequential(
            ConvBlock(in_channels, out_channels, 1, stride=1, padding=0),
            ConvBlock(out_channels, in_channels, 3, stride=1, padding=1)
        )


    def forward(self, x):
        return self.bottleneck(x)



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
    

class DarkNet19(nn.Module):

    config = [(32, 0), 'M', (64, 0), 'M', (128, 1), 'M', (256, 1), 'M', (512, 1), 'M', (1024, 2)]

    def __init__(self, in_channels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.darknet = self._build_darknet(in_channels)
    
    def _build_darknet(self, in_channels: int):
        blocks = []

        for layer in DarkNet19.config:
            if isinstance(layer, tuple):
                out_channels, n_bottleneck = layer
                blocks.append(DarkNetConvBlock(in_channels, out_channels, n_bottleneck))

                in_channels = out_channels
            else:
                blocks.append(nn.MaxPool2d(2, 2))


        return nn.Sequential(*blocks)
        

    def forward(self, x):
        return self.darknet(x)


class YOLOv2(nn.Module):
    def __init__(self, in_channels: int, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.darknet = DarkNet19(in_channels)

    def forward(self, x):
        return self.darknet(x)
    


if __name__ == "__main__":
    x = torch.randn(4, 3, 416, 416)

    model = YOLOv2(3)
    out = model(x)

    print(out.shape)