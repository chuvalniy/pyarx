import torch.nn as nn
import torch
from utils import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self):
        super(YoloLoss, self).__init__()

        self.n_boxes = 2
        self.n_classes = 20
        self.n_cells = 7

        self.mse = nn.MSELoss(reduction='sum')

        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        iou_b1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_b2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_b1.unsqueeze(0), iou_b2.unsqueeze(0)], dim=0)
        iou_maxes, best_box = torch.max(ious, dim=0)
        exists_box = target[..., 20].unsqueeze(3)

        # Box coordinate loss
        box_predictions = exists_box * (
                best_box * predictions[..., 26:30]
                + (1 - best_box) * predictions[..., 21:25]
        )
        box_targets = exists_box * target[..., 21:25]

        # Вторая часть box coordinate, где берем корень из W и H
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_targets[..., 2:4] = torch.sqrt(box_targets[..., 2:4])

        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_targets, end_dim=-2)
        )

        # Object loss
        pred_box = (
                best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21]
        )
        object_loss = self.mse(
            torch.flatten(exists_box * pred_box),
            torch.flatten(exists_box * target[..., 20:21])
        )

        # No-object loss
        no_object_loss = self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        no_object_loss += self.mse(
            torch.flatten((1 - exists_box) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - exists_box) * target[..., 20:21], start_dim=1)
        )

        # Class loss
        class_loss = self.mse(
            torch.flatten(exists_box * pred_box[..., :20], end_dim=-2),
            torch.flatten(exists_box * target[..., :20], end_dim=-2)
        )

        loss = self.lambda_coord * box_loss + object_loss + self.lambda_noobj * no_object_loss + class_loss
