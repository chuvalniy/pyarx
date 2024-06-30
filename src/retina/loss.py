import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """
    arXiv: https://arxiv.org/pdf/1708.02002v2
    """

    def __init__(self, gamma: int = 2, alpha: float = 0.5, eps: float = 1e-6):
        """
        :param gamma: Modulating factor (for curve)
        :param alpha: Weight for classes
        :param eps: Small value for logarithm to prevent division by zero.
        """
        super(FocalLoss, self).__init__()

        self.gamma = gamma
        self.alpha = alpha

        self.eps = eps

    def forward(self, y_pred, y_true):
        """

        :param y_pred: Predictions logits of shape (batch_size, n_classes)
        :param y_true: True classes (batch_size, )
        """
        batch_size, n_classes = y_pred.shape

        probs = F.softmax(y_pred, dim=-1)

        # One-hot encoding for true labels (batch_size, ) => (batch_size, n_classes)
        y_true_one_hot = F.one_hot(y_true, n_classes).float()

        pt = torch.sum(probs * y_true_one_hot, dim=-1)

        # Alpha parameter should be adjusted for positive & negative classes before application.
        alpha_t = self.alpha * y_true_one_hot + (1 - self.alpha) * (1 - y_true_one_hot)
        alpha_t = alpha_t.gather(1, y_true.unsqueeze(1)).squeeze(1)

        focal_weight = (1 - pt) ** self.gamma

        loss = -alpha_t * focal_weight * torch.log(pt + self.eps)
        return loss.mean()
