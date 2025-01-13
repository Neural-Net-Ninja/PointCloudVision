__all__ = ['TverskyLoss']

from typing import Optional

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
                 alpha: float = 0.5,
                 beta: float = 0.5,
                 reduction: str = "mean",
                 label_smoothing: Optional[float] = None):
        super().__init__(apply_softmax=apply_softmax, label_smoothing=label_smoothing, reduction=reduction)
        self.ignore_index = ignore_index
        self.epsilon = epsilon
        self.alpha = alpha
        self.beta = beta

    def forward(self,
                input: torch.Tensor,
                target: torch.Tensor,
                point_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Computes Tversky loss.

        :param input: The prediction tensor. Must have shape :math:`(N, C)` where `N = number of points`, and
            `C = number of classes`.
        :type input: torch.Tensor
        :param target: The target tensor. Must have shape :math:`(N)` where each element is a class index of integer
            type (label encoding).
        :type target: torch.Tensor
        :param point_weight: Manual rescaling weight given to each point. Must have shape :math:`(N)` where
            `N = number of points`. Defaults to `None`, which means that the point-wise losses are not rescaled.
        :type point_weight: torch.Tensor, optional
        :return: Scalar loss value.
        :rtype: torch.Tensor
        """
        input, target = self.flatten(input, target, self.ignore_index)
        if self.apply_softmax:
            input = F.softmax(input, dim=1)
        target = self.smooth_label(target, input.size(1))
        num_classes = input.size(1)
        losses = []

        for c in range(num_classes):
            target_c = (target == c).float()
            input_c = input[:, c]

            t_p = (input_c * target_c).sum()
            f_p = ((1 - target_c) * input_c).sum()
            f_n = (target_c * (1 - input_c)).sum()
            tversky = (t_p + self.epsilon) / (t_p + self.alpha * f_p + self.beta * f_n + self.epsilon)

            losses.append(1 - tversky)

        losses = torch.stack(losses)
        loss = losses.mean()

        if point_weight is not None:
            loss = point_weight * loss

        return self._reduce_loss(loss)