import torch
import torch.nn as nn
from vgg.model import VGG
import itertools
import math

from loss import SSDLoss


class DefaultBoxes:
    def __init__(self, fig_size, anchor_boxes, aspect_ratios):
        # Create default boxes for all feature maps at once
        # Output shape (n_feature_maps, n_boxes, 4) => (cx, cy, scaled_h, scaled_w)
        self.fig_size = fig_size
        self.aspect_ratios = aspect_ratios
        self.anchor_boxes = anchor_boxes

        self._s_min = 0.2
        self._s_max = 0.9
        self.scales = self._calculate_scales()

        self.boxes = self.generate_default_boxes()

    def _calculate_scales(self):
        scales = []
        for i in range(1, 6):  # Number of feature maps
            s_k = self._s_min + (self._s_max - self._s_min) / (6 - 1) * (i - 1)
            scales.append(s_k)

        return scales

    def generate_default_boxes(self):
        # Every scale is responsible for a single feature map, so len(scales) == n_feature_maps

        default_boxes = []
        for idx, scale in enumerate(self.scales):

            # Some feature maps require 4 or 6 aspect ratios for a single default box.
            n_anchor_boxes = self.anchor_boxes[idx]

            # Generate default boxes for every width & height pixel
            for i, j in itertools.product(range(self.fig_size[idx]), repeat=2):
                f_k = self.fig_size[idx]

                cx = (i + 0.5) / f_k
                cy = (j + 0.5) / f_k
                for a_r in self.aspect_ratios[:n_anchor_boxes]:
                    w = scale * math.sqrt(a_r)
                    h = scale / math.sqrt(a_r)

                    default_boxes.append((cx, cy, w, h))

        return torch.tensor(default_boxes)


class ClassifierSSD(nn.Module):
    """
    arXiv: https://arxiv.org/abs/1512.02325
    """

    def __init__(self, in_channels, n_anchors, n_classes):
        super(ClassifierSSD, self).__init__()

        self.out_channels = (n_classes + 4) * n_anchors
        self.conv = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1)

    def forward(self, x):
        batch_size = x.shape[0]

        x = self.conv(x)
        x = x.view(batch_size, -1, 24)
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

        self.clf_head_1 = ClassifierSSD(1024, 6, 20)
        self.clf_head_2 = ClassifierSSD(512, 6, 20)
        self.clf_head_3 = ClassifierSSD(256, 6, 20)
        self.clf_head_4 = ClassifierSSD(256, 4, 20)
        self.clf_head_5 = ClassifierSSD(256, 4, 20)

        self.default_boxes = DefaultBoxes(
            anchor_boxes=[6, 6, 6, 4, 4],
            fig_size=[20, 10, 5, 3, 1],
            aspect_ratios=[1, 1, 2, 3, 1 / 2, 1 / 3]
        ).boxes

    def forward(self, x):
        x = self.vgg_conv(x)

        x = self.conv6(x)
        x = self.conv7(x)
        clf_out_1 = self.clf_head_1(x)

        x = self.conv8_1(x)
        x = self.conv8_2(x)
        clf_out_2 = self.clf_head_2(x)

        x = self.conv9_1(x)
        x = self.conv9_2(x)
        clf_out_3 = self.clf_head_3(x)

        x = self.conv10_1(x)
        x = self.conv10_2(x)
        clf_out_4 = self.clf_head_4(x)

        x = self.conv11_1(x)
        x = self.conv11_2(x)
        clf_out_5 = self.clf_head_5(x)

        clf_outputs = torch.cat([
            clf_out_1,
            clf_out_2,
            clf_out_3,
            clf_out_4,
            clf_out_5
        ], dim=1)

        predictions = self._associate_boxes(clf_outputs)
        return predictions

    def _calculate_scale(self, k, m):
        scale = 0.2 + (0.9 - 0.2) / (m - 1) * (k - 1)
        return scale

    def _associate_boxes(self, x):
        pred_cx = self.default_boxes[:, 0] + self.default_boxes[:, 2] * x[:, :, 20]
        pred_cy = self.default_boxes[:, 1] + self.default_boxes[:, 3] * x[:, :, 21]
        pred_w = self.default_boxes[:, 2] + torch.exp(x[:, :, 22])
        pred_h = self.default_boxes[:, 3] + torch.exp(x[:, :, 23])

        pred_boxes = torch.stack([pred_cx, pred_cy, pred_w, pred_h], dim=-1)
        outputs = torch.cat([x[:, :, :20], pred_boxes], dim=-1)
        return outputs


if __name__ == '__main__':
    model = SSD()
    criterion = SSDLoss()

    x = torch.randn(8, 3, 300, 300)
    y = torch.randn(8, 5)
    out = model(x)

    loss = criterion(out, y)
