import torch
from torch.distributions.poisson import Poisson
import matplotlib.pyplot as plt

# Create a Poisson distribution with rate parameter 10
poisson = Poisson(10)

# Generate a set of 1000 points using Poisson disk sampling
points = poisson.sample((1000, 2))

# Plot the points
plt.scatter(points[:, 0], points[:, 1])
plt.show()