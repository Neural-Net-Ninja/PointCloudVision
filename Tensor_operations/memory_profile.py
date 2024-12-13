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

# Training loop with memory profiling
num_epochs = 5
for epoch in range(num_epochs):
    start_time = time.time()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.cuda(), target.cuda()

        # Forward pass
        output = model(data)
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

    end_time = time.time()
    print(f'Epoch [{epoch+1}/{num_epochs}] completed in {end_time - start_time:.2f} seconds')

print("Training completed.")

def pre_allocate_memory(self, batch_size: Optional[int] = None,
                        neighborhood_size: Optional[int] = None,
                        input_dim: Optional[int] = None,
                        ignore_oom_warnings: bool = False) -> bool:
    """
    Preallocates memory buffers of maximum size by executing a forward and backward pass with a batch of maximum
    sequence length. This can help to avoid unexpected out-of-memory errors and improve performance by reducing the
    need for memory reallocation during the actual training and prediction.

    :param batch_size: Batch size to be used for pre-allocation.
    :type batch_size: int
    :param neighborhood_size: Number of points in the neighborhood.
    :type neighborhood_size: int
    :param input_dim: Dimension of the input features.
    :type input_dim: int
    :param ignore_oom_warnings: If True, will not halt training on OOM error during pre-allocation.
    :type ignore_oom_warnings: bool
    :return: True if memory allocation is successful, False otherwise.
    :rtype: bool
    """
    if (self.input_size == 'variable'
            and isinstance(batch_size, int)
            and isinstance(neighborhood_size, int)
            and isinstance(input_dim, int)):

        torch.cuda.empty_cache()

        # Initiate dummy input data for memory pre-allocation
        training_coords = torch.randn(batch_size * neighborhood_size, input_dim, device=self.device)
        batch_indices = torch.tensor([0] * neighborhood_size + [1] * neighborhood_size, dtype=torch.long,
                                        device=self.device)
        neighborhood_sizes = torch.tensor([neighborhood_size, neighborhood_size], dtype=torch.long,
                                            device=self.device)
        try:
            self.train()
            self.zero_grad()
            output = self(training_coords, batch_indices, neighborhood_sizes)
            loss = output.mean()
            loss.backward()

            self.eval()
            with torch.no_grad():
                output = self(training_coords, batch_indices, neighborhood_sizes)

            return True  # Memory allocation successful
        except RuntimeError as e:
            if "out of memory" in str(e):
                if not ignore_oom_warnings:
                    logging.error("GPU out of memory during pre-allocation. Your batch size might lead to, "
                                    "OOM-Errors either reduce your batch-size or activate ignore_oom_warnings.")
                else:
                    logging.warning("Ignoring GPU out-of-memory error during pre-allocation as per user request.")
            else:
                raise e  # Re-throw the exception if it's not an out-of-memory error
    return False  # Memory allocation parameters are not valid