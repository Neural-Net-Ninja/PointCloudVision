import torch
from typing import Optional, List, Union

# Define mean function for second version
def mean(values, ignore_nan=False, empty=0):
    values = iter(values)
    if ignore_nan:
        values = filterfalse(lambda x: x != x, values)
    try:
        n = 1
        acc = next(values)
    except StopIteration:
        if empty == "raise":
            raise ValueError("Empty mean")
        return empty
    for n, v in enumerate(values, 2):
        acc += v
    if n == 1:
        return acc
    return acc / n

# Define _lovasz_grad for both versions
def _lovasz_grad(gt_sorted):
    p = len(gt_sorted)
    gts = gt_sorted.sum()
    intersection = gts - gt_sorted.float().cumsum(0)
    union = gts + (1 - gt_sorted).float().cumsum(0)
    jaccard = 1.0 - intersection / union
    if p > 1:
        jaccard[1:p] = jaccard[1:p] - jaccard[0:-1]
    return jaccard

# Second version of _lovasz_softmax_flat
def _lovasz_softmax_flat(probas, labels, classes="present", class_seen=None):
    if probas.numel() == 0:
        return probas * 0.0
    C = probas.size(1)
    losses = []
    class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
    for c in labels.unique():
        if class_seen is None or c in class_seen:
            fg = (labels == c).type_as(probas)
            if classes == "present" and fg.sum() == 0:
                continue
            if C == 1:
                if len(classes) > 1:
                    raise ValueError("Sigmoid output possible only with 1 class")
                class_pred = probas[:, 0]
            else:
                class_pred = probas[:, c]
            errors = (fg - class_pred).abs()
            errors_sorted, perm = torch.sort(errors, 0, descending=True)
            perm = perm.data
            fg_sorted = fg[perm]
            losses.append(torch.dot(errors_sorted, _lovasz_grad(fg_sorted)))
    return mean(losses)

# First version as a class method
class LovaszLoss:
    def __init__(self, weight=None):
        self.weight = weight

    def _lovasz_softmax_flat(self, probas, labels, classes="present", class_seen=None):
        if probas.numel() == 0:
            return probas * 0.0
        C = probas.size(1)
        losses = []
        class_to_sum = list(range(C)) if classes in ["all", "present"] else classes
        for c in class_to_sum:
            if class_seen is None or c in class_seen:
                fg = (labels == c).type_as(probas)
                if classes == "present" and fg.sum() == 0:
                    continue
                if C == 1:
                    if len(classes) > 1:
                        raise ValueError("Sigmoid output possible only with 1 class")
                    class_pred = probas[:, 0]
                else:
                    class_pred = probas[:, c]
                errors = (fg - class_pred).abs()
                errors_sorted, perm = torch.sort(errors, 0, descending=True)
                perm = perm.data
                fg_sorted = fg[perm]
                loss = torch.dot(errors_sorted, self._lovasz_grad(fg_sorted))
                if self.weight is not None:
                    class_weight = self.weight[torch.tensor(c)]
                    loss *= class_weight
                losses.append(loss)
        return self.mean(losses)

    def _lovasz_grad(self, gt_sorted):
        return _lovasz_grad(gt_sorted)

    def mean(self, values, ignore_nan=False):
        values = iter(values)
        if ignore_nan:
            values = filterfalse(lambda x: x != x, values)
        try:
            n = 1
            acc = next(values)
        except StopIteration:
            return torch.tensor(0.0)
        for n, v in enumerate(values, 2):
            acc += v
        return acc if n == 1 else acc / n

# Test function
def test_lovasz_softmax_flat():
    probas = torch.tensor([
        [0.7, 0.2, 0.1],  # Probabilities for 3 classes
        [0.1, 0.8, 0.1],
        [0.3, 0.3, 0.4],
        [0.2, 0.1, 0.7]
    ], dtype=torch.float)
    labels = torch.tensor([0, 1, 2, 2], dtype=torch.long)
    weight = None  # Test without weights first

    loss_module = LovaszLoss(weight=weight)
    
    loss1 = loss_module._lovasz_softmax_flat(probas, labels, classes="present")
    loss2 = _lovasz_softmax_flat(probas, labels, classes="present")
    
    print(f"Loss (First Version, classes='present'): {loss1.item():.4f}")
    print(f"Loss (Second Version, classes='present'): {loss2.item():.4f}")
    
    # Test with classes="all"
    loss1_all = loss_module._lovasz_softmax_flat(probas, labels, classes="all")
    loss2_all = _lovasz_softmax_flat(probas, labels, classes="all")
    
    print(f"Loss (First Version, classes='all'): {loss1_all.item():.4f}")
    print(f"Loss (Second Version, classes='all'): {loss2_all.item():.4f}")

if __name__ == "__main__":
    test_lovasz_softmax_flat()