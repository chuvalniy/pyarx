from utils import intersection_over_union
from collections import defaultdict


def mean_average_precision(y_pred, y_true, iou_threshold):
    """
    :param y_pred: Predictions for bounding boxes (n_pred_boxes, 5)
    :param y_true: (n_true_boxes, 4)
    :param iou_threshold: IoU threshold to filter bounding boxes.
    """
    # Create thresholds via sorted confidences
    thresholds = sorted(y_pred[:, 1])

    scores = []
    for thrsh in thresholds:
        true_to_pred = {}
        for i, pred_bbox in enumerate(y_pred):
            if pred_bbox[:1] < thrsh:
                continue

            for j, true_bbox in enumerate(y_true):
                iou = intersection_over_union(pred_bbox, true_bbox)
                if iou >= iou_threshold:
                    true_to_pred[j] = i

        false_negatives = len(y_true) - len(true_to_pred)
        false_positives = len(y_pred) - len(true_to_pred)
        true_positives = len(true_to_pred)

        precision = true_positives / (true_positives + false_positives)
        recall = true_positives / (true_positives + false_negatives)

        scores.append([precision, recall])



