import torch.nn as nn
import torch
from src.utils.iou import intersection_over_union



class YOLOv2Loss(nn.Module):
    def __init__(
            self, 
            lambda_noojb: float, 
            lambda_prior: float, 
            lambda_coord: float,
            lambda_obj: float, 
            lambda_cls: float
        ):
        super(YOLOv2Loss, self).__init__()

        self.n_iter = 0 
        self.mse_loss = nn.MSELoss()

        self.lambda_noobj = lambda_noojb
        self.lambda_prior = lambda_prior
        self.lambda_coord = lambda_coord
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls


    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        """Calculate YOLOv2 Loss.

        Parameters
        ----------
        y_true : torch.Tensor (B, S, S, 125)
            Ground truth. 
        y_pred : torch.Tensr (B, S, S, 125)
            Model predictions.
        """
        batch_size, cell_size = y_true.shape[:2]

        _y_true = y_true.reshape(batch_size, cell_size, cell_size, 5, 25)
        _y_pred = y_pred.reshape(batch_size, cell_size, cell_size, 5, 25)

        _y_pred[..., 21:23] = torch.sigmoid(_y_pred[..., 21:23])
        _y_pred[..., 23:25] = torch.exp(_y_pred[..., 23:25])

        has_object = _y_true[..., 20:21]
        no_object = 1 - has_object


        # Confidence loss
        conf_loss = self.lambda_obj * self.mse_loss(
            torch.flatten(has_object * _y_pred[..., 20:21]),
            torch.flatten(has_object * _y_true[..., 20:21])
        )

        # Classification loss
        cls_loss = self.lambda_cls * self.mse_loss(
            torch.flatten(has_object * _y_pred[..., :20]),
            torch.flatten(has_object * _y_true[..., :20])
        )

        # Bounding box loss
        box_loss =  self.lambda_coord * self.mse_loss(
            torch.flatten(has_object * _y_pred[..., 21:25]),
            torch.flatten(has_object * _y_true[..., 21:25])
        )

        # No-object loss
        noobj_loss = no_object * self.lambda_noobj * (-y_pred[..., 20:21])**2

        return conf_loss + cls_loss + box_loss + noobj_loss        
