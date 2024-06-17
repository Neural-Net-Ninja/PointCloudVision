import unittest
import torch
from Tversky_loss import TverskyLoss

class TestTverskyLoss(unittest.TestCase):
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

if __name__ == '__main__':
    unittest.main()