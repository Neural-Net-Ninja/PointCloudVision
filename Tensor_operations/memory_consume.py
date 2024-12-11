import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
import time

# Define a simple model
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28 * 28)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# Load data
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset, batch_size=64, shuffle=True)

# Initialize model, loss function, and optimizer
model = SimpleModel().cuda()
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Example dimensions
B = 32     # Batch size
N = 10000  # Number of points
D = 3      # Number of dimensions

# Memory for points (float32)
points_memory = B * N * D * 4  # 4 bytes per float32

# Memory for batch_indices (long)
batch_indices_memory = B * N * 8  # 8 bytes per long

# Memory for labels (long)
labels_memory = B * N * 8  # 8 bytes per long

# Memory for point_cloud_sizes (long)
point_cloud_sizes_memory = B * 8  # 8 bytes per long

# Total memory for a single batch
single_batch_memory = points_memory + batch_indices_memory + labels_memory + point_cloud_sizes_memory

# Number of batches to pre-allocate for
num_batches = 10

# Total memory requirement
total_memory = single_batch_memory * num_batches

# Add memory for model parameters (example: 500 MB)
model_memory = 500 * 1024 * 1024

# Total memory including model parameters
total_memory += model_memory

# Calculate the number of elements needed
num_elements = total_memory // 4  # 4 bytes per float32

# Memory profiling before pre-allocation
mem_before_prealloc = torch.cuda.memory_allocated()

# Create a tensor with the calculated number of elements
pre_alloc_tensor = torch.empty((num_elements,), device='cuda')
del pre_alloc_tensor
torch.cuda.empty_cache()

# Memory profiling after pre-allocation
mem_after_prealloc = torch.cuda.memory_allocated()
mem_consumed_prealloc = mem_after_prealloc - mem_before_prealloc

print(f'Memory Consumed by Pre-allocation: {mem_consumed_prealloc / 1024 ** 2:.2f} MB')

# Training loop with memory profiling
num_epochs = 5
for epoch in range(num_epochs):
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        # Memory profiling before forward pass
        mem_before = torch.cuda.memory_allocated()

        # Forward pass
        output = model(data)

        # Memory profiling after forward pass
        mem_after = torch.cuda.memory_allocated()
        mem_consumed = mem_after - mem_before

        loss = criterion(output, target)

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Memory profiling
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
            print(f'Memory Allocated: {torch.cuda.memory_allocated() / 1024 ** 2:.2f} MB')
            print(f'Memory Reserved: {torch.cuda.memory_reserved() / 1024 ** 2:.2f} MB')
            print(f'Memory Consumed by Forward Pass: {mem_consumed / 1024 ** 2:.2f} MB')

    end_time = time.time()
    print(f'Epoch [{epoch+1}/{num_epochs}] completed in {end_time - start_time:.2f} seconds')

print("Training completed.")