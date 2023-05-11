import torch
import pointops

# pointops grouping taked in (reference_index, key, coord, with_xyz=True)

def grouping (input: torch.Tensor, group_idx: torch.Tensor) -> torch.Tensor:
    """
    Grouping operation for a 3D tensor. The grouping is done along the last dimension of the tensor.
    :param input: input tensor of shape (B, N, C) where B is the batch size, N is the number of points and C is the
    number of channels
    :param group_idx: tensor of shape (B, N, K) where B is the batch size, N is the number of points and K is the
    number of neighbors
    :return: tensor of shape (B, N, K, C) where B is the batch size, N is the number of points, K is the number of
    neighbors and C is the number of channels
    """
    B, N, C = input.size()
    _, _, K = group_idx.size()

    # Reshape input to (B, N, 1, C)
    input = input.unsqueeze(2)
    # Reshape group_idx to (B, N, K, 1)
    group_idx = group_idx.unsqueeze(3)

    # Gather input points based on group_idx
    grouped_points = torch.gather(input, 1, group_idx.expand(-1, -1, -1, C))
    return grouped_points

B = 3  # Batch size
N = 10  # Number of elements
C = 5  # Number of channels
K = 2  # Number of neighbors

# input of shape (B, N, C) where B is the batch size, N is the number of points and C is the number of channels
input = torch.randn(3, 10, 4)

# group_idx of shape (B, N, K) where B is the batch size, N is the number of points and K is the number of neighbors
group_idx = torch.randint(N, size=(B, N, K))

print("input:__", "\n", input.shape)
print("group_idx:__", "\n", group_idx.shape)

grouped_points = grouping(input, group_idx)
print("grouped points:__", "\n", grouped_points.shape)



    