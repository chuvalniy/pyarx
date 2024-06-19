import torch
import torch.nn as nn
from vgg.model import VGG
import itertools
import math


class DefaultBoxes:
    def __init__(self, scales, anchor_boxes, fig_size, aspect_ratios):
        """

        """
        # Create default boxes for all feature map at once
        # Output shape (n_feature_maps, n_boxes, 4) => (cx, cy, scaled_h, scaled_w)
        self.scales = scales
        self.anchor_boxes = anchor_boxes  # n_feature_maps, n_anchor_boxes
        self.fig_size = fig_size  # n_feature_maps, fig_size
        self.aspect_ratios = aspect_ratios  # n_feature_maps, n_aspect_ratios

    def generate_default_boxes(self):

        # Every scale is responsible for a single feature map, so len(scales) == n_feature_maps
        all_default_boxes = []
        for idx, scale in enumerate(self.scales):
            default_boxes = []
            for i, j in itertools.product(range(self.fig_size[idx]), repeat=2):
                f_k = self.fig_size

                cx = (i + 0.5) / f_k
                cy = (j + 0.5) / f_k
                for a_r in self.aspect_ratios[idx]:
                    w = scale * math.sqrt(a_r)
                    h = scale / math.sqrt(a_r)


class ClassifierSSD(nn.Module):
    def __init__(self, in_channels, n_anchors, n_classes):
        super(ClassifierSSD, self).__init__()

        out_channels = n_anchors * (n_classes + 4)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0):
        super(ConvBlock, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.relu(self.conv(x))
        return x


class SSD(nn.Module):
    def __init__(self):
        super(SSD, self).__init__()
        vgg_layers = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512]
        vgg = VGG(3, 100, vgg_layers)
        self.vgg_conv = vgg.conv_blocks

        self.conv6 = ConvBlock(512, 1024, padding=1)
        self.conv7 = ConvBlock(1024, 1024, kernel_size=1)

        self.conv8_1 = ConvBlock(1024, 256, kernel_size=1)
        self.conv8_2 = ConvBlock(256, 512, stride=2, padding=1)

        self.conv9_1 = ConvBlock(512, 128, kernel_size=1)
        self.conv9_2 = ConvBlock(128, 256, stride=2, padding=1)

        self.conv10_1 = ConvBlock(256, 128, kernel_size=1)
        self.conv10_2 = ConvBlock(128, 256)

        self.conv11_1 = ConvBlock(256, 128, kernel_size=1)
        self.conv11_2 = ConvBlock(128, 256)

        # TODO:
        self.clf_head_1 = ClassifierSSD(1024, 4, 20)
        self.clf_head_2 = ClassifierSSD(512, 6, 20)
        self.clf_head_3 = ClassifierSSD(256, 6, 20)
        self.clf_head_4 = ClassifierSSD(256, 6, 20)
        self.clf_head_5 = ClassifierSSD(256, 4, 20)

    def forward(self, x):
        x = self.vgg_conv(x)

        x = self.conv6(x)
        x = self.conv7(x)
        clf_1 = self.clf_head_1(x)
        print(clf_1.shape)

        x = self.conv8_1(x)
        x = self.conv8_2(x)
        clf_2 = self.clf_head_2(x)
        print(clf_2.shape)

        x = self.conv9_1(x)
        x = self.conv9_2(x)
        clf_3 = self.clf_head_3(x)
        print(clf_3.shape)

        x = self.conv10_1(x)
        x = self.conv10_2(x)
        clf_4 = self.clf_head_4(x)
        print(clf_4.shape)

        x = self.conv11_1(x)
        x = self.conv11_2(x)
        clf_5 = self.clf_head_5(x)
        print(clf_5.shape)

        return x

    def _calculate_scale(self, k, m):
        scale = 0.2 + (0.9 - 0.2) / (m - 1) * (k - 1)
        return scale


if __name__ == '__main__':
    model = SSD()

    x = torch.randn(1, 3, 300, 300)
    out = model(x)
