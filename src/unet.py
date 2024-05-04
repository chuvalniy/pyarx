import torch.nn as nn
import torch


class UNetConv(nn.Module):
    def __init__(self, in_channels, out_channels, require_max_pool=True):
        super(UNetConv, self).__init__()

        self.require_max_pool = require_max_pool

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.max_pool = nn.MaxPool2d(2, 2)

    def forward(self, x):
        out = self.relu1(self.conv1(x))
        pre_out = self.relu2(self.conv2(out))  # We need this in decoder as encoder_weights

        if self.require_max_pool:
            out = self.max_pool(pre_out)
        else:
            out = pre_out.clone()

        return out, pre_out


class UNetDeconv(nn.Module):
    def __init__(self, in_channels, out_channels, require_upconv=True):
        super(UNetDeconv, self).__init__()
        self.require_upconv = require_upconv

        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3)

        self.relu1 = nn.ReLU(inplace=True)
        self.relu2 = nn.ReLU(inplace=True)

        self.upconv = nn.ConvTranspose2d(out_channels, out_channels // 2, kernel_size=2, stride=2)

    def _crop(self, x, encoder_feature_map):
        """
        Crops the encoder feature map tensor along the height and width dimensions to align with input tensor.

        Example:
        x shape (batch_size, channels, 56, 56)
        encoder_weights shape (batch_size, channels, 64, 64)

        Crop encoder_weights to have shape as in x (batch_size, channels, 56, 56)
        """

        _, _, decoder_height, decoder_width = x.size()

        encoder_weights_crop = encoder_feature_map[:, :, :decoder_height, :decoder_width]
        return encoder_weights_crop

    def forward(self, x, encoder_feature_map):
        enc_weights = self._crop(x, encoder_feature_map)
        x = torch.cat([enc_weights, x], dim=1)

        out = self.relu1(self.conv1(x))
        out = self.relu2(self.conv2(out))

        if self.require_upconv:
            out = self.upconv(out)

        return out


class UNet(nn.Module):
    _encoder_channels = [64, 128, 256, 512, 1024]
    _decoder_channels = [512, 256, 128, 64]

    def __init__(self, in_channels=3):
        super(UNet, self).__init__()

        self.encoder = self._create_block(in_channels, UNet._encoder_channels, is_encoder=True)
        self.decoder = self._create_block(1024, UNet._decoder_channels, is_encoder=False)

        self.deconv = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.conv = nn.Conv2d(64, 2, kernel_size=1)

        self._encoder_feature_maps = []

    def _create_block(self, in_channels, channels, is_encoder=True):
        layers = []

        for i, out_channels in enumerate(channels):
            require_pool = i != len(channels) - 1  # Use MaxPool if it's not the last layer of the block.

            if is_encoder:
                block = UNetConv(in_channels, out_channels, require_max_pool=require_pool)
            else:
                block = UNetDeconv(in_channels, out_channels, require_upconv=require_pool)

            layers.append(block)

            in_channels = out_channels

        return layers

    def forward(self, x):
        for layer in self.encoder:
            x, pre_out = layer(x)  # We take pre_out as final output from encoder due to x has 14x14 after MaxPool.
            self._encoder_feature_maps.append(pre_out)
        self._encoder_feature_maps.pop(-1)

        x = self.deconv(x)

        for layer in self.decoder:
            feature_map = self._encoder_feature_maps.pop(-1)
            x = layer(x, feature_map)

        x = self.conv(x)
        return x


if __name__ == '__main__':
    x = torch.randn(1, 3, 572, 572)

    net = UNet()
    out = net(x)

    print(out.shape)
