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
        """
        Forward pass of the Tversky Loss.

        :param inputs: Predictions tensor of shape (N, C, H, W), where C = number of classes.
        :param targets: Ground truth tensor of shape (N, H, W) with class indices.
        :returns: Computed Tversky Loss.
        """
        # Convert targets to one-hot encoding
        targets_one_hot = F.one_hot(targets, num_classes=inputs.shape[1]).permute(0, 3, 1, 2).float()

        # Flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets_one_hot = targets_one_hot.view(-1)

        # True positives, false positives, and false negatives
        TP = (inputs * targets_one_hot).sum()
        FP = ((1 - targets_one_hot) * inputs).sum()
        FN = (targets_one_hot * (1 - inputs)).sum()

        # Calculate Tversky score
        Tversky = (TP + self.smooth) / (TP + self.alpha * FP + self.beta * FN + self.smooth)

        # Compute Tversky loss
        loss = 1 - Tversky

        # Apply reduction
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss