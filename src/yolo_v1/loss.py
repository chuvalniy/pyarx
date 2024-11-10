import torch
import torch.nn as nn
from src.utils.iou import intersection_over_union


class YOLOv1Loss(nn.Module):
    def __init__(self, lambda_noobj: float = 0.5, lambda_coord: float = 5.0, eps: float = 1e-6, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.lambda_noobj = lambda_noobj
        self.lambda_coord = lambda_coord 

        self.eps = eps

        self.mse_loss = nn.MSELoss()

    def forward(self, y_true, y_pred):
        """Calculate YOLOv1 loss.

        Parameters
        ----------
        y_true : torch.Tensor(B, S, S, 25)
            Ground truth.
        y_pred : torch.Tensor(B, S, S, 30)
            Model predictions.
        """
        box_iou1 = intersection_over_union(y_pred[..., 21:25], y_true[..., 21:25])
        box_iou2 = intersection_over_union(y_pred[..., 26:30], y_true[..., 21:25])

        concat_iou = torch.concat([box_iou1, box_iou2], dim=-1)
        box_ids = torch.argmax(concat_iou, dim=-1).unsqueeze(-1)

        has_object = y_true[..., 20:21]
        no_object = 1 - has_object

        # Find resbonsible box for cell prediction
        responsible_box = box_ids * y_pred[..., 20:25] + (1 - box_ids) * y_pred[..., 25:30]

        # Bounding Box Loss
        responsible_box[..., 3:5] = (
            torch.sign(responsible_box[..., 3:5]) * 
            torch.sqrt(torch.abs(responsible_box[..., 3:5]) + self.eps)
        )


        target_box = torch.concat(
            [y_true[..., 21:23], torch.sqrt(y_true[..., 23:25])],
            dim=-1
        )
        box_loss = self.mse_loss(
            torch.flatten(has_object * responsible_box[..., 1:5], end_dim=-2), 
            torch.flatten(has_object * target_box, end_dim=-2)  # Redundant 'has_object'?
        )

        # Object Confiedence Loss
        conf_loss = self.mse_loss(
            torch.flatten(has_object * responsible_box[..., 0:1], end_dim=-2),
            torch.flatten(has_object * y_true[..., 20:21], end_dim=-2)  # Redundant 'has_object'?
        )

        # Classification Loss
        cls_loss = self.mse_loss(
            torch.flatten(has_object * y_pred[..., 0:20], end_dim=-2),
            torch.flatten(has_object * y_true[..., 0:20], end_dim=-2)  # Redundant 'has_object'?
        )

        object_loss = self.lambda_coord * (box_loss + conf_loss + cls_loss)

        # No-object Loss
        no_object_loss = self.lambda_noobj * self.mse_loss(
            torch.flatten(no_object * responsible_box[..., 0:1], end_dim=-2),
            torch.flatten(no_object * y_true[..., 20:21], end_dim=-2)  # Redundant 'has_object'?
        )

        # Total Loss
        loss = object_loss + no_object_loss
        return loss