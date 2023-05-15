import torch
#import pointops

# pointops knn_query takes in (neighbours, coord, offset)

# neighbours = 8
# coord.size() = torch.Size([2, 1024, 3])
# offset = torch.Size([2])
# offset = tensor([1024, 2048])

# returns indices, distances
# indices = torch.Size([2, 8])

indices = torch.tensor([[  0,  10,  20,  30,  40,  50,  60,  70],
                       [  1,  11,  21,  31,  41,  51,  61,  71]])

print("indices:__", "\n", indices.shape)


def KNNQueryNaive(ctx, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor = None) -> Tuple[torch.Tensor]:
    """
    KNN Indexing
    input: nsample: int32, Number of neighbor
            xyz: (b, n, 3) coordinates of the features
            new_xyz: (b, m, 3) centriods
        output: idx: (b, m, nsample)
    """
    if new_xyz is None:
        new_xyz = xyz
    b, m, _ = new_xyz.size()
    n = xyz.size(1)

    '''
    idx = torch.zeros(b, m, nsample).int().cuda()
    for i in range(b):
        dist = pairwise_distances(new_xyz[i, :, :], xyz[i, :, :])
        [_, idxs] = torch.sort(dist, dim=1)
        idx[i, :, :] = idxs[:, 0:nsample]
    '''


    new_xyz_repeat = new_xyz.repeat(1, 1, n).view(b, m * n, 3)
    xyz_repeat = xyz.repeat(1, m, 1).view(b, m * n, 3)
    dist = (new_xyz_repeat - xyz_repeat).pow(2).sum(dim=2).view(b, m, n)
    
    [_, idxs] = torch.sort(dist, dim=2)
    
    # KNNQueryExclude
    # idx = idxs[:, :, 1:nsample+1].int()
    idx = idxs[:, :, 0:nsample].int()
    
    return idx



def KNNQueryExclude(ctx, nsample: int, xyz: torch.Tensor, new_xyz: torch.Tensor = None) -> Tuple[torch.Tensor]:
    """
    KNN Indexing
    input: nsample: int32, Number of neighbor
            xyz: (b, n, 3) coordinates of the features
            new_xyz: (b, m, 3) centriods
        output: new_features: (b, m, nsample)
    """
    if new_xyz is None:
        new_xyz = xyz
    b, m, _ = new_xyz.size()
    n = xyz.size(1)

    '''
    idx = torch.zeros(b, m, nsample).int().cuda()
    for i in range(b):
        dist = pairwise_distances(new_xyz[i, :, :], xyz[i, :, :])
        [_, idxs] = torch.sort(dist, dim=1)
        idx[i, :, :] = idxs[:, 0:nsample]
    '''

    new_xyz_repeat = new_xyz.repeat(1, 1, n).view(b, m * n, 3)
    xyz_repeat = xyz.repeat(1, m, 1).view(b, m * n, 3)
    dist = (new_xyz_repeat - xyz_repeat).pow(2).sum(dim=2).view(b, m, n)

    [_, idxs] = torch.sort(dist, dim=2)
    idx = idxs[:, :, 1:nsample+1].int()

    return idx