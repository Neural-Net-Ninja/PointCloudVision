# Can you generate a simple deep learning example using Pytorch?

import torch
import torch.nn as nn   # Neural Network                                    
import torch.nn.functional as F  # Neural Network Functions 
import torch.optim as optim  # Optimization

# Define the Neural Network
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square convolution
        # kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(16 * 6 * 6, 120)  # 6*6 from image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        # If the size is a square you can only specify a single number
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

net = Net()
print(net)

# Define the Loss Function
criterion = nn.CrossEntropyLoss()

# Define the Optimizer
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# Train the Network
for epoch in range(2):  # loop over the dataset multiple times
    
    running_loss = 0.0 # Initialize the loss        
    for i, data in enumerate(trainloader, 0): # Loop over the data
        
        # Get the inputs
        inputs, labels = data

        # Zero the parameter gradients
        optimizer.zero_grad()
        
        # Forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        
        # Update the weights
        optimizer.step()
        
        # Print statistics
        running_loss += loss.item()

model.eval()
with torch.no_grad():
    for data in testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
print('Accuracy of the network on the 10000 test images: %d %%' % ( 100 * correct / total))

# Save the model
PATH = './cifar_net.pth'
torch.save(net.state_dict(), PATH)