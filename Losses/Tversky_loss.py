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

    def setUp(self):
        self.inputs = torch.tensor([[[[0.7, 0.2], [0.4, 0.6]]]], dtype=torch.float32)
        self.targets = torch.tensor([[[[1, 0], [1, 1]]]], dtype=torch.long)
        self.epsilon = 1e-5  # Small value to account for floating-point arithmetic errors

    def test_initialization(self):
        loss = TverskyLoss()
        self.assertEqual(loss.alpha, 0.5)
        self.assertEqual(loss.beta, 0.5)
        self.assertEqual(loss.smooth, 1.0)
        self.assertEqual(loss.reduction, 'mean')
        self.assertIsNone(loss.class_weights)
        self.assertFalse(loss.dynamic_focus)

    def test_forward_pass(self):
        loss = TverskyLoss()
        result = loss(self.inputs, self.targets)
        self.assertTrue(torch.is_tensor(result))
        self.assertGreater(result.item(), 0)

    def test_class_weights(self):
        class_weights = torch.tensor([0.5, 2.0])
        loss = TverskyLoss(class_weights=class_weights)
        result_with_weights = loss(self.inputs, self.targets)
        result_without_weights = TverskyLoss()(self.inputs, self.targets)
        self.assertNotEqual(result_with_weights.item(), result_without_weights.item())

    def test_dynamic_focus(self):
        loss = TverskyLoss(dynamic_focus=True)
        initial_alpha = loss.alpha
        initial_beta = loss.beta
        loss(self.inputs, self.targets)  # Trigger dynamic focus adjustment
        self.assertNotEqual(initial_alpha, loss.alpha)
        self.assertNotEqual(initial_beta, loss.beta)

    def test_adjust_focus_increase_alpha(self):
        loss = TverskyLoss(dynamic_focus=True)
        # Simulate a case where precision is lower than recall
        loss.adjust_focus(TP=50, FP=100, FN=30)
        self.assertLess(loss.alpha, 0.5)
        self.assertGreater(loss.beta, 0.5)

    def test_adjust_focus_increase_beta(self):
        loss = TverskyLoss(dynamic_focus=True)
        # Simulate a case where recall is lower than precision
        loss.adjust_focus(TP=50, FP=30, FN=100)
        self.assertGreater(loss.alpha, 0.5)
        self.assertLess(loss.beta, 0.5)

    @patch('Tversky_loss.logging')
    def test_logging(self, mock_logging):
        loss = TverskyLoss()
        loss.forward(self.inputs, self.targets)
        mock_logging.getLogger.assert_called_with('TverskyLoss')
        mock_logging.getLogger('TverskyLoss').info.assert_called()

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

    def adjust_focus(self, TP, FP, FN):
        """
        Dynamically adjusts alpha and beta based on the performance of the last epoch.
        """
        if TP + FP == 0 or TP + FN == 0:  # Avoid division by zero
            return
        precision = TP / (TP + FP)
        recall = TP / (TP + FN)
        if precision < recall:
            self.alpha *= 0.95
            self.beta *= 1.05
        else:
            self.alpha *= 1.05
            self.beta *= 0.95
        self.alpha = max(min(self.alpha, 0.9), 0.1)
        self.beta = max(min(self.beta, 0.9), 0.1)
        self.epoch += 1
        self.logger.info(f'Adjusted alpha to {self.alpha}, beta to {self.beta} for epoch {self.epoch}')