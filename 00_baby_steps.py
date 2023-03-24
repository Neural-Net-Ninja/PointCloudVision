import numpy as np
import torch

# Input (temp, rainfall, humidity)
inputs = np.array([[73, 67, 43],
                   [91, 88, 64],
                   [87, 134, 58],
                   [102, 43, 37],
                   [69, 96, 70]], dtype='float32')

# Targets (apples, oranges)
targets = np.array([[56, 70],
                    [81, 101],
                    [119, 133],
                    [22, 37],
                    [103, 119]], dtype='float32')

# Convert inputs and targets to tensors
inputs = torch.from_numpy(inputs)
targets = torch.from_numpy(targets)
print('inputs', inputs)
print('targets', targets)

# Weights and biases
w = torch.randn(2, 3, requires_grad=True)
b = torch.randn(2, requires_grad=True)
print('w', w)
print('b', b)

def model(x):
    return x @ w.t() + b

# Generate predictions 
preds = model(inputs)
print('preds', preds)

# Compare with targets
print('targets', targets)

# MSE loss
def mse(t1, t2):
    diff = t1 - t2
    return torch.sum(diff * diff) / diff.numel()

# Compute loss
loss = mse(preds, targets)
print('loss', loss)

# Compute gradients
loss.backward()

# Gradients for weights
print('w', w)
print('w.grad', w.grad)