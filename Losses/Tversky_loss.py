__all__ = ['TverskyLoss']

from typing import Optional, Literal

import torch
import torch.nn.functional as F

from .base_loss import SegmentationLoss


class TverskyLoss(SegmentationLoss):
    """
    Tversky loss function for imbalanced datasets.

    The Tversky loss is defined as:

    .. math::

        \mathcal{L}_T = 1 - \frac{TP + \epsilon}{TP + \alpha \cdot FP + \beta \cdot FN + \epsilon}

    where:
    - \( TP \) (True Positives),
    - \( FP \) (False Positives),
    - \( FN \) (False Negatives),
    - \( \alpha \) controls the weight of false positives (default: 0.3),
    - \( \beta \) controls the weight of false negatives (default: 0.7),
    - \( \epsilon \) is a smoothing factor to avoid division by zero.

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
    :param reduction: Specifies the reduction to aggregate the loss values of a batch and multiple classes:
        `"mean"` | `"sum"`. `"none"`: no reduction will be applied, `"mean"`: the mean of the output is taken, `"sum"`:
        the output will be summed (default = `"mean"`).
    :type reduction: str, optional
    :param weight:A manual rescaling weight given to each class. If given, has to be a Tensor of shape `(C)`, where
        `C = number of classes`.
    :type weight: torch.Tensor
    :type weight: Optional[torch.Tensor], optional
    :param label_smoothing: A float in [0.0, 1.0]. Specifies the amount of smoothing when computing the loss,
        where 0.0 means no smoothing. Defaults to 0.
    :type label_smoothing: float, optional
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
            `C = number of classes`. If :attr:`apply_softmax` is `True`, each element is expected to be a logit value.
            Otherwise, each element is expected to be a class probability.
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
        valid_mask = (target != self.ignore_index).float()

        prediction = prediction.reshape(prediction.shape[0], num_classes, -1)
        one_hot_target = one_hot_target.reshape(one_hot_target.shape[0], num_classes, -1)
        valid_mask = valid_mask.reshape(valid_mask.shape[0], -1)

        TP = torch.sum(prediction * one_hot_target * valid_mask.unsqueeze(1), dim=2)
        FP = torch.sum(prediction * (1 - one_hot_target) * valid_mask.unsqueeze(1), dim=2)
        FN = torch.sum((1 - prediction) * one_hot_target * valid_mask.unsqueeze(1), dim=2)

        tversky_index = (TP + self.epsilon) / (TP + self.alpha * FP + self.beta * FN + self.epsilon)
        loss = 1 - tversky_index

        if self.weight is not None:
            loss = loss * self.weight.view(1, -1)

        if self.ignore_index is not None:
            loss = loss[:, :self.ignore_index]

        if point_weight is not None:
            loss = loss * point_weight

        return self._reduce_loss(loss)
