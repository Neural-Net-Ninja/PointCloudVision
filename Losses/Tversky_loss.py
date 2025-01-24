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
    :param alpha: Weight for false positives. Defaults to 0.3.
    :type alpha: float, optional
    :param beta: Weight for false negatives. Defaults to 0.7.
    :type beta: float, optional
    :param reduction: Specifies the reduction to apply to the output: `"mean"` | `"sum"`.
        Defaults to `"mean"`.
    :type reduction: str, optional
    """

    def __init__(self,
                 apply_softmax: bool = True,
                 ignore_index: int = 255,
                 epsilon: float = 1.0,
                 alpha: float = 0.3,
                 beta: float = 0.7,
                 reduction: Literal["mean", "sum"] = "mean",
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
        valid_mask = (target != self.ignore_index).float().unsqueeze(1)

        prediction = prediction * valid_mask
        one_hot_target = one_hot_target * valid_mask

        TP = torch.sum(prediction * one_hot_target, dim=0)
        FP = torch.sum(prediction * (1 - one_hot_target), dim=0)
        FN = torch.sum((1 - prediction) * one_hot_target, dim=0)

        tversky_index = (TP + self.epsilon) / (TP + self.alpha * FP + self.beta * FN + self.epsilon)
        loss = 1 - tversky_index

        if self.weight is not None:
            loss = loss * self.weight

        if point_weight is not None:
            point_weight = point_weight.unsqueeze(1).expand_as(prediction)
            loss = loss * point_weight

        return self._reduce_loss(loss)