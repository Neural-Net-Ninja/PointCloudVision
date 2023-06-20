import torch


def poisson_sampling(points: torch.Tensor, indices: torch.Tensor, num_points: int, radius: float) -> torch.Tensor:
    """
    Generates a Poisson disk sample of `num_points` points from the input `points` using the given `indices` and
    `radius`. The function first computes the minimum and maximum coordinates of the input points, the size of each grid
    cell, and the dimensions of the grid. It then initializes the grid with -1 to indicate that each cell is empty, and
    the list of active points with a random point. The function iteratively selects a random active point, generates
    candidate points in the vicinity of the active point, rejects candidate points that are too close to existing
    points, and adds a random candidate point to the list of active points. The function also adds the sampled point to
    the grid and converts the sampled indices back to a tensor.

    :param points: Input point cloud of shape (N, 3), where `N = number of points`.
    :type points: torch.Tensor
    :param indices: Indices of the points to sample of shape (M,), where `M = number of points to sample`.
    :type indices: torch.Tensor
    :param num_points: Number of points to sample.
    :type num_points: int
    :param radius: Minimum distance between points.
    :type radius: float
    :return: Sample indices of shape (num_points,).
    :rtype: torch.Tensor
    """
    # Compute the minimum and maximum coordinates of the input points.
    min_coords = torch.min(points, dim=0).values
    max_coords = torch.max(points, dim=0).values

    # Compute the size of each grid cell.
    cell_size = radius / math.sqrt(points.shape[1])

    # Compute the dimensions of the grid.
    grid_dims = ((max_coords - min_coords) / cell_size).ceil().int()

    # Initialize the grid with -1 to indicate that each cell is empty.
    grid = torch.full(grid_dims.tolist(), -1, dtype=torch.long)

    # Initialize the list of active points with a random point.
    active_points = [torch.randint(0, indices.shape[0], (1,)).item()]
    sampled_indices = [indices[active_points[0]].item()]

    # Add the first sampled point to the grid.
    grid_coords = ((points[sampled_indices[0]] - min_coords) / cell_size).int()
    grid[grid_coords[0], grid_coords[1], grid_coords[2]] = 0

    while len(active_points) > 0 and len(sampled_indices) < num_points:
        # Select a random active point.
        idx = torch.randint(0, len(active_points), (1,)).item()
        point_idx = active_points[idx]
        point = points[indices[point_idx]]

        # Generate candidate points in the vicinity of the active point.
        candidates = torch.randn((30, points.shape[1]), dtype=points.dtype, device=points.device) * radius + point
        candidates = candidates[(candidates >= min_coords) & (candidates <= max_coords)]

        # Reject candidate points that are too close to existing points.
        distances = torch.cdist(candidates, points[sampled_indices])
        valid_mask = torch.all(distances > radius, dim=1)
        candidates = candidates[valid_mask]

        # If there are no valid candidates, remove the active point from the list.
        if len(candidates) == 0:
            active_points.pop(idx)
        else:
            # Select a random candidate point and add it to the list of active points.
            idx = torch.randint(0, len(candidates), (1,)).item()
            active_points.append(len(sampled_indices))
            sampled_indices.append(indices[(points == candidates[idx]).all(dim=1)].item())
            # Add the sampled point to the grid.
            grid_coords = ((candidates[idx] - min_coords) / cell_size).int()
            grid[grid_coords[0], grid_coords[1], grid_coords[2]] = len(sampled_indices) - 1

    # Convert the sampled indices back to a tensor.
    sampled_indices = torch.tensor(sampled_indices, dtype=torch.long)

    return sampled_indices