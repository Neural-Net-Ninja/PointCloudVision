


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