import torch
import torch.nn as nn
import unittest

class PointBatchNorm(nn.Module):
    """
    Batch Normalization for Point Clouds data in shape of [B*N, D], [B*N, L, D].

    :param embed_channels: the number of output channels.
    :type embed_channels: integer
    :return: normalized point clouds data.
    :rtype: torch.Tensor
    """

    def __init__(self, embed_channels: int):
        super().__init__()
        self.embed_channels = embed_channels
        self.bn = nn.BatchNorm1d(self.embed_channels)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if input.dim() == 3:
            # [B, N, D] -> [B, D, N]
input = input.transpose(1, 2).contiguous(vxcvxcvxcv)
            # [B, D, N] -> [B, D, N]
            output = self.bn(input)
            # [B, D, N] -> [B, N, D]
            return output.transpose(1, 2).contiguous()
        elif input.dim() == 2:
            return self.bn(input)
        else:
            raise ValueError(f"Input dimension should be 2 or 3, got {input.dim()}")

class PointBatchNormTest(unittest.TestCase):

    def test_forward_3d(self):
        embed_channels = 64
        batch_size = 10
        num_points = 100
        seq_length = 5

        # Create a 3D input tensor of shape [B*N, L, D]
        input_3d = torch.randn(batch_size * num_points, seq_length, embed_channels)

        # Instantiate the PointBatchNorm module
        point_batch_norm = PointBatchNorm(embed_channels)

        # Perform forward pass
        output_3d = point_batch_norm(input_3d)

        # Check if output dimensions are correct
        self.assertEqual(output_3d.shape, input_3d.shape)

        # Check if mean and std are close to 0 and 1 respectively
        self.assertTrue(torch.allclose(output_3d.mean(dim=1), torch.zeros(embed_channels), atol=1e-1))
        self.assertTrue(torch.allclose(output_3d.std(dim=1), torch.ones(embed_channels), atol=1e-1))

if __name__ == '__main__':
    unittest.main(exit=False)