import torch
import torch.nn as nn
import torch.nn.functional as F

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.5, beta=0.5, smooth=1.0, reduction='mean'):
        """
        Initializes the Tversky Loss.

        :param alpha: Controls the penalty for false positives.
        :param beta: Controls the penalty for false negatives.
        :param smooth: Smoothing factor to avoid division by zero.
        :param reduction: Specifies the reduction to apply to the output: 'none' | 'mean' | 'sum'
        """
        super(TverskyLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        self.reduction = reduction

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