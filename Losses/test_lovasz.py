


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