import unittest
from parameterized import parameterized
import numpy as np
import torch
import torch.nn.functional as F

from pcnn.losses import TverskyLoss


class TestTverskyLoss(unittest.TestCase):
    @parameterized.expand([
        ("mean", 0.5, 0.5, 1.0, np.array([0.25, 0.2, 0.1]).mean()),
    ])
    def test_probabilistic_predictions(self, reduction: str,
                                       alpha: float,
                                       beta: float,
                                       epsilon: float,
                                       expected_loss: np.ndarray):
        """
        Tests that the Tversky loss is computed correctly when the predictions are probabilities between zero and one.
        """
        prediction = torch.as_tensor(
            [
                [0.5, 0.25, 0.25],
                [0.8, 0.1, 0.1],
                [0.6, 0.3, 0.1]
            ],
            dtype=torch.float
        )

        target = torch.as_tensor([1, 0, 2])

        loss_module = TverskyLoss(apply_softmax=False, alpha=alpha, beta=beta, epsilon=epsilon, reduction=reduction)

        actual_loss = loss_module(prediction, target).numpy()

        np.testing.assert_allclose(expected_loss, actual_loss, atol=1e-07)


if __name__ == '__main__':
    unittest.main(exit=False)