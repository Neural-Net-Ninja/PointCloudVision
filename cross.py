import torch


class label_smooth_loss(torch.nn.Module):
    def __init__(self, num_classes, smoothing=0.0):
        super(label_smooth_loss, self).__init__()
        eps = smoothing / num_classes
        self.negative = eps
        self.positive = (1 - smoothing) + eps

    def forward(self, pred, target):
        pred = pred.log_softmax(dim=1)
        true_dist = torch.zeros_like(pred)
        true_dist.fill_(self.negative)
        true_dist.scatter_(1, target.data.unsqueeze(1), self.positive)
        return torch.sum(-true_dist * pred, dim=1).mean()


prediction = torch.as_tensor(
            [
                [-1000, -1000, 1000],
                [1000, -1000, -1000],
                [-1000, 1000, -1000]
            ],
            dtype=torch.float
        )

target = torch.as_tensor([2, 0, 1])


loss1 = label_smooth_loss(num_classes=3, smoothing=0.1)
loss2 = torch.nn.CrossEntropyLoss(apply_softmax=True, label_smoothing=0.1)


print(loss1(prediction, target), loss2(prediction, target))

