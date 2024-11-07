import torch
import torch.nn as nn

class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()
        n