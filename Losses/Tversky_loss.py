import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0, reduction='mean', class_weights=None, dynamic_focus=False):
        """
        Enhanced Tversky Loss with class weights, logging, and dynamic focus.

        :param alpha: Controls the penalty for false positives.
        :param beta: Controls the penalty for false negatives.
        :param smooth: Smoothing factor to avoid division by zero.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        :param class_weights: Optional tensor of weights for each class.
        :param dynamic_focus: If True, adjusts alpha and beta dynamically based on epoch performance.
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction
        self.class_weights = class_weights
        self.dynamic_focus = dynamic_focus
        self.logger = logging.getLogger('TverskyLoss')
        self.epoch = 0  # Track the current epoch for dynamic adjustments

    def forward(self, inputs, targets):
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()

        if self.class_weights is not None:
            targets_one_hot = targets_one_hot * self.class_weights.view(1, -1, 1, 1)

        inputs_flat = inputs.view(-1)
        targets_flat = targets_one_hot.view(-1)

        TP = (inputs_flat * targets_flat).sum()
        FP = ((1 - targets_flat) * inputs_flat).sum()
        FN = (targets_flat * (1 - inputs_flat)).sum()

        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)
        loss = 1 - Tversky

        if self.dynamic_focus:
            self.adjust_focus(TP, FP, FN)

        if self.reduction == 'mean':
            loss = loss.mean()
        elif self.reduction == 'sum':
            loss = loss.sum()

        self.logger.info(f'Epoch {self.epoch}: Tversky Loss = {loss.item()}')
        return loss