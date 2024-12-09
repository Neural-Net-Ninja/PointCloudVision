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