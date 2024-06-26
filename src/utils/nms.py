from iou import intersection_over_union
import torch


def non_maximum_suppression(x, iou_threshold=0.5):
    """
    Leave bounding box with the highest confidence and suppress other bounding boxes with the high overlap

    :param x: scores for bounding boxes (batch_size, [cls, conf, cx, cy, h, w])
    :param iou_threshold: IoU threshold for suppressing bounding box
    """

    sorted_tensor, sorted_indices = torch.sort(x[:, 0], descending=True)
    sorted_x = x[sorted_indices]




if __name__ == '__main__':
    x = torch.tensor([
        [0.5, 0.75, 0.69, 0.1, 0.1],
        [0.7, 0.2, 0.25, 0.1, 0.11],
        [0.2, 0.2, 0.25, 0.21, 0.13]
    ])

    bboxes = non_maximum_suppression(x)
