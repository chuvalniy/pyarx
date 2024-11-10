
import torch

class YOLOv1Postprocessor:
    def __init__(self, input_size: tuple[int, int] = (448, 448), threshold: float = 0.5):
        """
        Parameters
        ----------
        input_size : tuple[int, int], optional
            _description_, by default (448, 448)
        threshold : float, optional
            Object confidence threshold, by default 0.5
        """

        self.height = input_size[0]
        self.width = input_size[1]

        self.threshold = threshold

    def _format_box(self, box: torch.Tensor) -> torch.Tensor:
        """Format box to have absolute coordinates.

        Parameters
        ----------
        box : torch.Tensor(B, N, N, 4)
            Tensor with bounding boxes.

        Returns
        -------
        torch.Tensor(B, N, N, 4)
            Bounding boxes with absolute coordinates.
        """
        n_cell = box.shape[1]

        height_step = self.height // n_cell
        width_step = self.width // n_cell

        x_offset = torch.arange(0, self.height, height_step).repeat(n_cell, 1).float()
        y_offset = torch.arange(0, self.width, width_step).repeat(n_cell, 1).t().float()
        
        new_box = box.clone()
        new_box[..., 0] = x_offset[None, ...] + new_box[..., 0] * width_step
        new_box[..., 1] = y_offset[None, ...] + new_box[..., 1] * height_step
        new_box[..., 2] = self.width * new_box[..., 2]
        new_box[..., 3] = self.height * new_box[..., 3]

        return new_box

    def __call__(self, x: torch.Tensor) -> torch.Tensor:
        """Post-process model output to get parsed predictions.

        Parameters
        ----------
        x : torch.Tensor(B, S, S, 30)
            Raw model output.

        Returns
        -------
        torch.Tensor(B, M, 6)
            Parsed model output of M boxes with xywh format.
        """
        batch_size = x.shape[0]

        box1 = self._format_box(x[..., 21:25])
        box2 = self._format_box(x[..., 26:30])

        best_cls, best_cls_ids = torch.max(x[..., :20], dim=-1, keepdim=True)
        
        box1_conf = x[..., 20:21] * best_cls
        box2_conf = x[..., 25:26] * best_cls

        best_conf, best_box_ids = torch.max(
            torch.concat([box1_conf, box2_conf], dim=-1), 
            dim=-1, 
            keepdim=True
        )
        best_boxes = best_box_ids * box2 + (1 - best_box_ids) * box1
        
        output = torch.concat(
            [best_boxes, best_conf, best_cls_ids],
            dim=-1
        ).reshape(batch_size, -1, 6)  # From (B, N, N, 6) to (B, N*N, 6)

        return output