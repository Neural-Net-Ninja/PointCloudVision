__all__ = ['TverskyLoss']

from typing import Optional, Literal

import torch
import torch.nn.functional as F

from .base_loss import SegmentationLoss


class TverskyLoss(SegmentationLoss):
    """
    Tversky loss function for imbalanced datasets.

    :param apply_softmax: Whether the predictions passed to the loss function are logits that need to be converted to
        probabilities by applying Softmax activation function. Defaults to `True`.
    :type apply_softmax: bool, optional
    :param ignore_index: Specifies a target value that is ignored and does not contribute to the input gradient.
    :type ignore_index: int, optional
    :param epsilon: Smoothing factor to avoid division by zero. Defaults to 1.0.
    :type epsilon: float, optional
    :param alpha: Weight for false positives. Defaults to 0.5.
    :type alpha: float, optional
    :param beta: Weight for false negatives. Defaults to 0.5.
    :type beta: float, optional
    :param reduction: Specifies the reduction to apply to the output: `"none"` | `"mean"` | `"sum"`.
        Defaults to `"mean"`.
    :type reduction: str, optional
    """

    def __init__(self,
                 apply_softmax: bool = True,
                 ignore_index: int = 255,
                 epsilon: float = 1.0,
                 alpha: float = 0.3,
                 beta: float = 0.7,
                 reduction: Literal["mean", "sum", "none"] = "mean",
                 weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.0):
        super().__init__(apply_softmax=apply_softmax, label_smoothing=label_smoothing, reduction=reduction)
        self.ignore_index = ignore_index
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta
        self.weight = weight

    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                point_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes Tversky loss.

        :param prediction: The prediction tensor. Must have shape :math:`(N, C)` where `N = number of points`, and
            `C = number of classes`.
        :type prediction: torch.Tensor
        :param target: The target tensor. Must have shape :math:`(N)` where each element is a class index of integer
            type (label encoding).
        :type target: torch.Tensor
        :param point_weight: Manual rescaling weight given to each point. Must have shape :math:`(N)` where
            `N = number of points`. Defaults to `None`, which means that the point-wise losses are not rescaled.
        :type point_weight: torch.Tensor, optional
        :return: Scalar loss value.
        :rtype: torch.Tensor
        """
        if self.apply_softmax:
            prediction = F.softmax(prediction, dim=1)

        num_classes = prediction.size(1)
        one_hot_target = self.smooth_label(target, num_classes)
        valid_mask = (target != self.ignore_index).long()

        losses = []
        for c in range(num_classes):
            if c == self.ignore_index:
                continue
            pred_c = prediction[:, c]
            pred_c = pred_c.reshape(pred_c.shape[0], -1)
            target_c = one_hot_target[:, c]
            target_c = target_c.reshape(target_c.shape[0], -1)
            valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

            TP = torch.sum(torch.mul(pred_c, target_c) * valid_mask, dim=1)
            FP = torch.sum(torch.mul(pred_c, 1 - target_c) * valid_mask, dim=1)
            FN = torch.sum(torch.mul(1 - pred_c, target_c) * valid_mask, dim=1)

            tversky_index = (TP + self.epsilon) / (TP + self.alpha * FP + self.beta * FN + self.epsilon)
            loss = 1 - tversky_index

            if self.weight is not None:
                loss *= self.weight[c]

            losses.append(loss)

        loss = torch.stack(losses)

        if point_weight is not None:
            loss = loss * point_weight

        return self._reduce_loss(loss)
