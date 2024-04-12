__all__ = ['LovaszLoss']

from typing import Optional, Literal
from itertools import filterfalse

import torch
from torch import Tensor
from .base_loss import SegmentationLoss


class LovaszLoss(SegmentationLoss):  # take in label smoothing and weight as parameters from models.py
    """
    Lovasz loss for segmentation task.

    :param class_seen: Class seen. Defaults to None.
    :type class_seen: Optional[int], optional
    :param per_image: If True, loss computed per each image and then averaged, else computed per whole batch
        Defaults to False.
    :type per_image: bool, optional
    :param ignore_index: Label that indicates ignored pixels (does not contribute to loss). Defaults to None.
    :type ignore_index: Optional[int], optional
    :param point_weight: Point weight. Defaults to 1.0.
    :type point_weight: float, optional
    """
    def __init__(self,
                 apply_softmax: bool = True,
                 class_seen: Optional[int] = None,
                 per_image: bool = False,
                 ignore_index: Optional[int] = None,
                 label_smoothing: float = 0.0,
                 reduction: Literal["mean", "sum", "none"] = "mean",
                 weight: Optional[torch.Tensor] = None):
        super().__init__(apply_softmax=apply_softmax, label_smoothing=label_smoothing, reduction=reduction)
        self.ignore_index = ignore_index
        self.per_image = per_image
        self.class_seen = class_seen
        self.weight = weight

    def _lovasz_softmax(self,
                        probas: Tensor,
                        labels: Tensor,
                        classes: str = "present",
                        class_seen: Optional[int] = None,
                        per_image: bool = False,
                        ignore: Optional[int] = None) -> Tensor:
        """
        Multi-class Lovasz-Softmax loss

        :param probas: Class probabilities at each prediction (between 0 and 1). Interpreted as binary (sigmoid)
            output with outputs of size [B, H, W].
        :type probas: torch.Tensor
        :param labels: Ground truth labels (between 0 and C - 1)
        :type labels: torch.Tensor
        :param classes: 'all' for all, 'present' for classes present in labels, or a list of classes
            to average. Defaults to "present".
        :type classes: str, optional
        : param class_seen: Class seen. Defaults to None.
        :type class_seen: Optional[int], optional
        :param per_image: Compute the loss per image instead of per batch. Defaults to False.
        :type per_image: bool, optional
        :param ignore: Void class labels. Defaults to None.
        :type ignore: Optional[int], optional
        """
        if per_image:
            losses = []
            for prob, lab in zip(probas, labels):
                flat_probas, flat_labels = self._flatten_probas(prob.unsqueeze(0), lab.unsqueeze(0), ignore)
                loss = self._lovasz_softmax_flat(flat_probas, flat_labels, classes=classes)
                losses.append(loss)
            loss = self.mean(losses)
        else:
            flat_probas, flat_labels = self._flatten_probas(probas, labels, ignore)
            loss = self._lovasz_softmax_flat(flat_probas, flat_labels, classes=classes, class_seen=class_seen)

        return loss

    def _lovasz_softmax_flat(self, probas, labels, classes="present", class_seen=None):
        """
        Multi-class Lovasz-Softmax loss

        :param probas: [P, C] Class probabilities at each prediction (between 0 and 1)
        :type probas: torch.Tensor
        :param labels: [P] Tensor, ground truth labels (between 0 and C - 1)
        :type labels: torch.Tensor
        :param classes: 'all' for all, 'present' for classes present in labels, or a list of classes to average.
        :type classes: str, optional
        """
        if probas.numel() == 0:
            # only void pixels, the gradients should be 0
            return probas * 0.0
        C = probas.size(1)
        losses = []
        for c in labels.unique():
            if class_seen is None or c in class_seen:
                fg = (labels == c).type_as(probas)  # foreground for class c
                if classes == "present" and fg.sum() == 0:
                    continue
                if C == 1:
                    if len(classes) > 1:
                        raise ValueError("Sigmoid output possible only with 1 class")
                    class_pred = probas[:, 0]
                else:
                    class_pred = probas[:, int(c)]

                errors = (fg - class_pred).abs()
                errors_sorted, perm = torch.sort(errors, 0, descending=True)
                perm = perm.data
                fg_sorted = fg[perm]
                loss = torch.dot(errors_sorted, self._lovasz_grad(fg_sorted))

            # Apply the class weight
            if self.weight is not None:
                class_weight = self.weight[c]
                loss *= class_weight

            losses.append(loss)

        return self.mean(losses)

    def _flatten_probas(self, probas, labels, ignore=None):
        """
        Flattens predictions in the batch

        :param probas: [B, C, H, W] or [B, H, W] Variable, class probabilities at each prediction (between 0 and 1)
        :type probas: torch.Tensor
        :param labels: [B, H, W] Variable, ground truth labels (between 0 and C - 1)
        :type labels: torch.Tensor
        :param ignore: void class labels
        :type ignore: Optional[int], optional
        """
        if probas.dim() == 3:
            # assumes output of a sigmoid layer
            B, H, W = probas.size()
            probas = probas.view(B, 1, H, W)

        C = probas.size(1)
        probas = torch.movedim(probas, 1, -1)
        probas = probas.contiguous().view(-1, C)

        labels = labels.view(-1)
        if ignore is None:
            return probas, labels
        valid = labels != ignore
        vprobas = probas[valid]
        vlabels = labels[valid]
        return vprobas, vlabels

    def isnan(self, x):
        return x != x

    def mean(self, values, ignore_nan=False, empty=0):
        """
        Nan-mean compatible with generators.
        """
        values = iter(values)
        if ignore_nan:
            values = filterfalse(self.isnan, values)
        try:
            n = 1
            acc = next(values)
        except StopIteration:
            if empty == "raise":
                raise ValueError("Empty mean")
            return empty
        for n, v in enumerate(values, 2):
            acc += v
        if n == 1:
            return acc
        return acc / n

    def _lovasz_grad(self, gt_sorted):
        """
        Compute gradient of the Lovasz extension w.r.t sorted errors

        :param gt_sorted: [P] Variable, sorted ground truth errors
        :type gt_sorted: torch.Tensor
        """
        p = len(gt_sorted)
        gts = gt_sorted.sum()
        intersection = gts - gt_sorted.float().cumsum(0)
        union = gts + (1 - gt_sorted).float().cumsum(0)
        jaccard = 1.0 - intersection / union
        if p > 1:  # cover 1-pixel case
            jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
        return jaccard

    def forward(self,
                prediction: torch.Tensor,
                target: torch.Tensor,
                point_weight: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of the loss function.

        :param prediction: Prediction tensor of shape (N, C, H, W)
        :type prediction: torch.Tensor
        :param target: Target tensor of shape (N, H, W) or (N, C, H, W)
        :type target: torch.Tensor

        :returns: Loss value.
        :rtype: torch.Tensor
        """
        prediction = torch.nn.functional.softmax(prediction, dim=-1) if self.apply_softmax else prediction
        target = self.smooth_label(target, prediction.size(-1))

        loss = self._lovasz_softmax(
            prediction,
            target,
            class_seen=self.class_seen,
            per_image=self.per_image,
            ignore=self.ignore_index)

        if point_weight is not None:
            loss = point_weight * loss

        return self._reduce_loss(loss)
