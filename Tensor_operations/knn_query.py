import torch
import pointops

# pointops knn_query takes in (neighbours, coord, offset)

# neighbours = 8
# coord.size() = torch.Size([2, 1024, 3])
# offset = torch.Size([2])
# offset = tensor([1024, 2048])

# returns indices, distances
# indices = torch.Size([2, 8])