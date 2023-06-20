import torch
import unittest


def grouping(idx,  # shape (N, K)
             feat,  # shape (N, D)
             xyz,  # shape (N, 3)
             new_xyz=None,
             with_xyz=False):
    if new_xyz is None:
        new_xyz = xyz
    assert xyz.is_contiguous() and feat.is_contiguous()

    m, nsample, c = idx.shape[0], idx.shape[1], feat.shape[1]
    xyz = torch.cat([xyz, torch.zeros([1, 3]).to(xyz.device)], dim=0)  # shape (N, 3)
    feat = torch.cat([feat, torch.zeros([1, c]).to(feat.device)], dim=0)  # shape (N, D)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c)  # (N, K, D)

    if with_xyz:
        assert new_xyz.is_contiguous()
        mask = torch.sign(idx + 1)  #shape (N, K)
        un_sq = new_xyz.unsqueeze(1) # shape (N, 1, 3)
        grouped_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3)  # shape (N, K, 3)
        actual = grouped_xyz - un_sq  # (m, num_sample, 3)  # shape (N, K, 3)
        grouped_xyz = torch.einsum("n s c, n s -> n s c", grouped_xyz, mask)  # (N, K, 3)
        return torch.cat((grouped_xyz, grouped_feat), -1)
    else:
        return grouped_feat
    return grouped_points



idx = torch.tensor([[0, 1, 2], [1, 2, 3], [2, 3, 4], [3, 4, 5]])

feat = torch.tensor([[0, 0, 0, 10], [1, 1, 1, 10], [2, 2, 2, 10], [3, 3, 3, 10], 
                     [6, 6, 6, 11], [7, 7, 7, 11], [8, 8, 8, 11], [11, 11, 11, 11]])

xyz = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], 
                    [6, 6, 6], [7, 7, 7], [8, 8, 8], [11, 11, 11]])


output = grouping(idx, feat, xyz, with_xyz=True)
print("output:__", "\n", output, "\n", output.shape)