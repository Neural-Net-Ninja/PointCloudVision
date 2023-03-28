import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc1 = nn.Linear(784, 512)
        self.dropout1 = nn.Dropout(0.5) # dropout probability of 0.5
        self.fc2 = nn.Linear(512, 256)
        self.dropout2 = nn.Dropout(0.5) # dropout probability of 0.5
        self.fc3 = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1) # flatten input image
        x = torch.relu(self.fc1(x))
        x = self.dropout1(x) # apply dropout to the first fully connected layer
        x = torch.relu(self.fc2(x))
        x = self.dropout2(x) # apply dropout to the second fully connected layer
        x = self.fc3(x)
        return x

model = MyModel()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# train the model
model.train()
for epoch in range(num_epochs):
    for images, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

# evaluate the model
model.eval()
with torch.no_grad():
    correct = 0
    total = 0
    for images, labels in test_loader:
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy: {:.2f}%'.format(100 * correct / total))

# save the model
torch.save(model.state_dict(), 'model.ckpt')

# working with the saved model and inserting in the the new function.