from iou import intersection_over_union
import torch


def non_maximum_suppression(x, iou_threshold=0.5):
    """
    Leave bounding box with the highest confidence and suppress other bounding boxes with the high overlap

    :param x: scores for bounding boxes (batch_size, [cls, conf, cx, cy, h, w])
    :param iou_threshold: IoU threshold for suppressing bounding box
    """

    boxes = sorted(x, key=lambda i: i[1], reverse=True)

    nms_boxes = []
    while boxes:
        chosen_box = boxes.pop(0)

        boxes = [
            box
            for box in boxes
            if box[0] != chosen_box[0]
            or
            intersection_over_union(
                chosen_box[2:], box[2:]
            ) < iou_threshold
        ]

        nms_boxes.append(chosen_box)

    return nms_boxes


if __name__ == '__main__':
    x = torch.tensor([
        [1, 0.5, 0.1, 0.1, 0.3, 0.3],
        [1, 0.7, 0.1, 0.11, 0.3, 0.31],
        [0, 0.2, 0.2, 0.25, 0.21, 0.13]
    ])

    bboxes = non_maximum_suppression(x)
