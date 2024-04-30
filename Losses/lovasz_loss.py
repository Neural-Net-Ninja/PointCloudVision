"""
The Lovász hinge loss, or simply Lovász loss, is a loss function commonly used in the context of structured prediction 
problems, particularly in semantic image segmentation tasks. 
It extends the concept of the hinge loss, which is often used in binary classification tasks, to handle structured 
output spaces where the order of predictions matters, such as segmentation masks.

The Lovász loss is particularly suitable for optimizing binary classification tasks with partial labels or imbalanced
classes. It measures the difference between predicted and ground truth segmentation masks in terms of the intersection 
over union (IoU) metric, also known as the Jaccard index.


Ordered Segmentation Masks: Lovász loss operates on ordered segmentation masks, where the pixels are sorted based on 
their predicted scores or probabilities. This ordering is crucial for computing the loss and ensures that the loss is 
permutation invariant.
Differentiable: Lovász loss is differentiable almost everywhere, allowing it to be used seamlessly with gradient-based 
optimization algorithms like stochastic gradient descent (SGD).
Extension of Hinge Loss: The Lovász loss can be seen as an extension of the hinge loss, which is commonly used in binary
classification problems. Instead of penalizing misclassifications directly, it penalizes deviations from the ideal 
ordering of predictions.
Interpretation: The Lovász loss can be interpreted as the average disagreement between predicted and ground truth 
segmentation masks, measured by the IoU metric.
Optimization: It is optimized directly with respect to the predicted scores or probabilities, rather than being 
indirectly optimized through a surrogate loss like cross-entropy.
Robustness: Lovász loss is known for its robustness to label noise and partial annotations, making it suitable for 
scenarios where ground truth labels may be incomplete or imprecise.
Application: Lovász loss finds widespread application in tasks such as semantic image segmentation, where the goal is 
to assign a semantic label to each pixel in an image.
In summary, the Lovász loss offers a principled way to optimize structured prediction tasks, particularly in scenarios 
where traditional loss functions may not be well-suited due to the nature of the output space. Its flexibility, 
differentiability, and robustness make it a valuable tool in the machine learning toolbox for tasks involving structured
outputs.
"""


__all__ = ['LovaszLoss']


from typing import Optional, Literal
from itertools import filterfalse

import torch
from torch import Tensor
from .base_loss import SegmentationLoss


class LovaszLoss(SegmentationLoss):
    """
    Lovasz loss for segmentation task.

    :param class_seen: Class seen. Defaults to None.
    :type class_seen: Optional[int], optional
    :param per_point: If True, loss computed per each image and then averaged, else computed per whole batch
        Defaults to False.
    :type per_point: bool, optional
    :param ignore_index: Label that indicates ignored pixels (does not contribute to loss). Defaults to None.
    :type ignore_index: Optional[int], optional
    :param point_weight: Point weight. Defaults to 1.0.
    :type point_weight: float, optional
    """
    def __init__(self,
                 apply_softmax: bool = True,
                 class_seen: Optional[int] = None,
                 per_point: bool = False,
                 ignore_index: Optional[int] = None,
                 label_smoothing: float = 0.0,
                 reduction: Literal["mean", "sum", "none"] = "mean",
                 weight: Optional[torch.Tensor] = None):
        super().__init__(apply_softmax=apply_softmax, label_smoothing=label_smoothing, reduction=reduction)
        self.ignore_index = ignore_index
        self.per_point = per_point
        self.class_seen = class_seen
        self.weight = weight

    def _lovasz_softmax(self,
                        probas: Tensor,
                        labels: Tensor,
                        classes: str = "present",
                        class_seen: Optional[int] = None,
                        per_point: bool = False,
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
        :param per_point: Compute the loss per image instead of per batch. Defaults to False.
        :type per_point: bool, optional
        :param ignore: Void class labels. Defaults to None.
        :type ignore: Optional[int], optional
        """
        if per_point:
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
        class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
        for c in class_to_sum:
            if class_seen is None or c in class_seen:
                fg = (labels == c).type_as(probas)  # foreground for class c
                if classes == "present" and fg.sum() == 0:
                    continue
                if C == 1:
                    if len(classes) > 1:
                        raise ValueError("Sigmoid output possible only with 1 class")
                    class_pred = probas[:, 0]
                else:
                    class_pred = probas[:, c]

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

    def _flatten_probas(self, probas, labels, ignore = Optional[int] = None):
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
        
        :param values: An iterable of numbers or nan values
        :type values: Iterable
        :param ignore_nan: Ignore nan values in the computation
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
            per_point=self.per_point,
            ignore=self.ignore_index)

        if point_weight is not None:
            loss = point_weight * loss

        return self._reduce_loss(loss)



