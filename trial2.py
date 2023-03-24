

import torch
target = torch.tensor([0, 1, 2, 0, 1, 2])
preds = torch.tensor([0, 2, 1, 0, 0, 1])
f1 = torch.F1Score(task="multiclass", num_classes=3)
f1(preds, target)