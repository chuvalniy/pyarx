import torch.nn as nn
import torch


class Yolov2Loss(nn.Module):
    def __init__(self):
        super(Yolov2Loss, self).__init__()


    def forward(self, y_true, y_pred):
        # noobj loss
        # anchor and pred loss
        # bb loss
        