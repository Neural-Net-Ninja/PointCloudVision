


@parameterized.expand([
    ("mean", np.array((-np.log(1 / 3) * (1 / 3 + 1 / 3 + 1 / 3) / 3),)),
    ("sum", np.array(-np.log(1 / 3) * (1 / 3 + 1 / 3 + 1 / 3)),),
])
def test_all_true(self, reduction: str, expected_loss: np.ndarray):
    """
    Tests that the Lovasz loss is computed correctly when all predictions are correct.
    """

    prediction = torch.as_tensor(
        [
            [0, 0, 1],
            [1, 0, 0],
            [0, 1, 0]
        ],
        dtype=torch.float
    )

    target = torch.as_tensor([2, 0, 1])

    loss_module = LovaszLoss(reduction=reduction)

    actual_loss = loss_module(prediction, target).numpy()

    np.testing.assert_allclose(expected_loss, actual_loss)
    
@parameterized.expand([
        ("mean", np.array(0.0)),
        ("sum", np.array(0.0)),
])
def test_all_true_1(self, reduction: str, expected_loss: np.ndarray):
    """
    Tests that the Lovasz loss is zero for a perfect prediction.
    """

    prediction = torch.as_tensor(
        [
            [1, 0, 0],  # Predicted probabilities for sample 1
            [0, 1, 0],  # Predicted probabilities for sample 2
            [0, 0, 1]   # Predicted probabilities for sample 3
        ],
        dtype=torch.float
    )

    target = torch.as_tensor([0, 1, 2])  # True class labels

    loss_module = LovaszLoss(per_point=True, reduction=reduction)

    actual_loss = loss_module(prediction, target).numpy()

    np.testing.assert_allclose(expected_loss, actual_loss)
        
# rewritting lovasz loss functino input tensors to make them less sensitive to ordering.

def test_flatten_probas(self):
    probas = torch.randn(1, 3, 224, 224)  # Dummy probabilities
    labels = torch.empty(1, 224, 224, dtype=torch.long).random_(3)  # Dummy labels
    ignore = None  # Dummy ignore value

    # Call _flatten_probas method
    flattened_probas, flattened_labels = self.model._flatten_probas(probas, labels, ignore)

    # Check that result is a tensor
    self.assertIsInstance(flattened_probas, torch.Tensor)
    self.assertIsInstance(flattened_labels, torch.Tensor)

    # Check that the shapes are as expected
    self.assertEqual(flattened_probas.shape, (1*3*224*224,))  # The shape should be (B*C*H*W,)
    self.assertEqual(flattened_labels.shape, (1*224*224,))  # The shape should be (B*H*W,)
    
def test_mean(self):
    values = np.array([1, 2, 3, 4, 5])  # Dummy values
    ignore_nan = False
    empty = 0

    # Call mean method
    result = self.model.mean(values, ignore_nan, empty)

    # Check that result is a float
    self.assertIsInstance(result, float)

    # Check that the result is as expected
    self.assertEqual(result, np.mean(values))  # The result should be the mean of the values