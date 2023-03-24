import torch


class label_smooth_loss(torch.nn.Module):
    def __init__(self, num_classes=3, smoothing=0.1):
        super(label_smooth_loss, self).__init__()
        self.num_classes = num_classes
        self.negative = smoothing / self.num_classes
        self.positive = (1 - smoothing) + self.negative

    def forward(self, target):
        true_dist = torch.full((len(target), self.num_classes), fill_value=self.negative, device=target.device)
        return true_dist.scatter_(-1, target.unsqueeze(1), self.positive)


prediction = torch.as_tensor(
            [
                [0.5, 0.25, 0.25],
                [0.8, 0.1, 0.1],
                [0.6, 0.3, 0.1]
            ],
            dtype=torch.float
        )

target = torch.as_tensor([2, 0, 1])


print(target)

loss1 = label_smooth_loss(num_classes=3, smoothing=0.1)

target = loss1(target)
print("smooth target", target)


cross_entropy_loss = -target * torch.log(prediction)

print(cross_entropy_loss)

cross_entropy_loss[target == 0] = 0


cross_entropy_loss = cross_entropy_loss.sum(dim=-1)

print(cross_entropy_loss)


