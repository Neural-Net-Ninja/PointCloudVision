
import torch
import unittest

def grouping(idx,
             feat,
             xyz,
             new_xyz=None,
             with_xyz=False):
    if new_xyz is None:
        new_xyz = xyz
    assert xyz.is_contiguous() and feat.is_contiguous()
    m, nsample, c = idx.shape[0], idx.shape[1], feat.shape[1]
    xyz = torch.cat([xyz, torch.zeros([1, 3]).to(xyz.device)], dim=0)
    feat = torch.cat([feat, torch.zeros([1, c]).to(feat.device)], dim=0)
    grouped_feat = feat[idx.view(-1).long(), :].view(m, nsample, c)  # (m, num_sample, c)

    if with_xyz:
        assert new_xyz.is_contiguous()
        mask = torch.sign(idx + 1)
        grouped_xyz = xyz[idx.view(-1).long(), :].view(m, nsample, 3) - new_xyz.unsqueeze(1)  # (m, num_sample, 3)
        grouped_xyz = torch.einsum("n s c, n s -> n s c", grouped_xyz, mask)  # (m, num_sample, 3)
        return torch.cat((grouped_xyz, grouped_feat), -1)
    else:
        return grouped_feat

class TestGrouping(unittest.TestCase):
    def test_grouping_with_xyz(self):
        idx = torch.tensor([[0, 1, 2], [3, 4, 5]])
        feat = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
        xyz = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
        new_xyz = torch.tensor([[0, 0, 0], [1, 1, 1]])
        with_xyz = True

        expected_output = torch.tensor([[[0, 0, 0, 1, 2, 3], [1, 1, 1, 4, 5, 6], [2, 2, 2, 7, 8, 9]],
                                        [[2, 2, 2, 10, 11, 12], [3, 3, 3, 13, 14, 15], [4, 4, 4, 16, 17, 18]]])

        output = grouping(idx, feat, xyz, new_xyz, with_xyz)
        self.assertTrue(torch.all(torch.eq(output, expected_output)))

    def test_grouping_without_xyz(self):
        idx = torch.tensor([[0, 1, 2], [3, 4, 5]])
        feat = torch.tensor([[1, 2, 3], [4, 5, 6], [7, 8, 9], [10, 11, 12], [13, 14, 15], [16, 17, 18]])
        xyz = torch.tensor([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3], [4, 4, 4], [5, 5, 5]])
        with_xyz = False

        expected_output = torch.tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]],
                                        [[10, 11, 12], [13, 14, 15], [16, 17, 18]]])

        output = grouping(idx, feat, xyz, None, with_xyz)
        self.assertTrue(torch.all(torch.eq(output, expected_output)))

if __name__ == '__main__':
    unittest.main()

