import torch


class YOLOv2PostProcessor:
    def __init__(self):
        self.anchor_boxes = torch.randn(5, 2)

    # TODO: Consider make split for boxes
    def forward(self, predictions):

        # x y w h obj_cof, classes
        # make batch of K, 1 where cx = x_K / K
        _pred = predictions.clone().reshape(4, 13, 13, 125)
        
        # Test with pen & paper, not sure about this part
        x_offset = torch.arange(13).repeat(13, 1).float() / 13
        y_offset = torch.arange(13).repeat(13, 1).t().float() / 13

        idxs = torch.arange(0, 125, 25)
        _pred[..., idxs] = torch.sigmoid(_pred[..., idxs]) + x_offset[None, ..., None]
        _pred[..., idxs + 1] = torch.sigmoid(_pred[...,  idxs + 1]) + y_offset[None, ..., None]
        _pred[..., idxs + 2] = self.anchor_boxes[..., 0] * torch.exp(_pred[..., idxs + 2])
        _pred[..., idxs + 3] = self.anchor_boxes[..., 1] * torch.exp(_pred[...,  idxs + 3])

        return _pred