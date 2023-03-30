import torch
import torch.nn as nn
import torch.optim as optim

# Define the individual models
class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        self.fc1 = nn.Linear(10, 3)
        self.fc2 = nn.Linear(3, 2)
        
    def forward(self, x):
        x = self.fc1(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        return x

# Define the ensemble model
class Ensemble(nn.Module):
    def __init__(self, model1, model2):
        super(Ensemble, self).__init__()
        self.model1 = model1
        self.model2 = model2
        self.fc = nn.Linear(4, 2)
        
    def forward(self, x):
        out1 = self.model1(x)
        out2 = self.model2(x)
        out = torch.cat((out1, out2), dim=1)
        out = self.fc(out)
        return out

# Define the data and optimizer
data = torch.randn(100, 10)
labels = torch.randint(2, (100,))
optimizer = optim.SGD(model.parameters(), lr=0.1)

# Train the individual models
model1 = Model1()
model2 = Model2()
criterion = nn.CrossEntropyLoss()

for epoch in range(10):
    optimizer.zero_grad()
    output1 = model1(data)
    loss1 = criterion(output1, labels)
    loss1.backward()
    optimizer.step()

    optimizer.zero_grad()
    output2 = model2(data)
    loss2 = criterion(output2, labels)
    loss2.backward()
    optimizer.step()

# Create the ensemble model and test it
ensemble = Ensemble(model1, model2)
output = ensemble(data)
loss = criterion(output, labels)
print(loss.item())
