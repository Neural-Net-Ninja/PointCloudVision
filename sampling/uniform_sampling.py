import torch


def uniform_sampling(points: torch.Tensor, indices: torch.Tensor, num_points: int) -> torch.Tensor:
    """
    Uniformly samples `num_points` points from the input `points` using the given `indices`. The function first
    generates a grid of indices for each dimension, then computes the meshgrid of indices for all dimensions, and
    finally samples num_points points from the meshgrid using the given indices.
    
    :param points: Input point cloud of shape (N, D), where `N = number of points` and `D = feature dimensions`.
    :type points: torch.Tensor
    :param indices: Indices of the points to sample of shape (M,), where `M = number of points to sample`.
    :type indices: torch.Tensor
    :param num_points: Number of points to sample.
    :type num_points: int
    return: Indices of the sampled points of shape (num_points,).
    :rtype: torch.Tensor
    """
    # Compute the number of points to select from each dimension.
    num_points_per_dim = int(torch.ceil(torch.pow(num_points, 1/points.shape[1])))

    # Compute the stride for each dimension.
    ranges = [torch.max(points[:, i]) - torch.min(points[:, i]) for i in range(points.shape[1])]
    stride = torch.tensor(ranges) / (num_points_per_dim - 1)

    # Generate a grid of indices for each dimension.
    grid_indices = [torch.arange(torch.min(points[:, i]), torch.max(points[:, i]) + stride[i], stride[i])
                    for i in range(points.shape[1])]

    # Generate a meshgrid of indices for all dimensions.
    meshgrid = torch.meshgrid(grid_indices)

    # Reshape the meshgrid into a list of coordinates.
    coords = torch.stack([meshgrid[i].flatten() for i in range(points.shape[1])], dim=1)

    # Sample `num_points` points from the coordinates.
    sampled_indices = indices[torch.randperm(indices.shape[0])[:num_points]]
    sampled_points = coords[sampled_indices]

    return sampled_points