import torch.nn as nn
import torch
from utils import intersection_over_union


class SSDLoss(nn.Module):
    def __init__(self, alpha=1.0):
        super(SSDLoss, self).__init__()
        self.smooth_l1 = nn.SmoothL1Loss(reduction='none')
        self.cross_entropy = nn.CrossEntropyLoss(reduction='none')
        self.alpha = alpha

    def forward(self, predictions, targets):

        batch_size, n_boxes = predictions.size(0), predictions.size(1)

        # Positive boxes.
        targets_bbox = targets[:, None, 1:5]
        predictions_bbox = predictions[:, :, 20:24]

        ious = intersection_over_union(predictions_bbox, targets_bbox)

        positive_indices = ious > 0.5
        negative_indices = ious < 0.5

        # Localization loss.
        loc_loss = self.smooth_l1(predictions_bbox, targets_bbox.expand_as(predictions_bbox))
        loc_loss = loc_loss.sum(dim=-1).unsqueeze(-1)
        loc_loss = loc_loss * positive_indices.float()
        loc_loss = loc_loss.sum() / positive_indices.float().sum()

        # Classification loss.
        predictions_scores = predictions[..., :20]
        targets_cls = targets[:, 0].unsqueeze(1).expand_as(predictions_scores[..., 0])

        predictions_scores_flat = predictions_scores.view(-1, predictions_scores.size(-1))
        targets_cls_flat = targets_cls.reshape(-1)

        cls_loss = self.cross_entropy(predictions_scores_flat, targets_cls_flat)
        cls_loss = cls_loss.view(batch_size, n_boxes)

        pos_cls_loss = cls_loss * positive_indices.float()
        # TODO: Add hard negatives
        neg_cls_loss = cls_loss * negative_indices.float()

        conf_loss = (pos_cls_loss.sum() + neg_cls_loss.sum()) / (
                positive_indices.float().sum() + negative_indices.float().sum())

        # Total loss.
        loss = conf_loss + self.alpha * loc_loss
        return loss
