import torch
from torchvision.models import resnet18, ResNet18_Weights

model = resnet18(weights=ResNet18_Weights.DEFAULT)
dataset = torch.rand(1, 3, 64, 64)
labels = torch.rand(1, 1000)

#print(f'Model before learning:______:', model)
for key, value in model.named_parameters():
    print(f'model parameters key:______:\n {key} \n')
    print(f'model parameters value:______:\n {value} \n')



for epoch in range(3):

    # forward pass
    prediction = model(dataset)

    # Calculating loss
    loss = (prediction - labels).sum()

    # backward pass
    loss.backward()

    optim = torch.optim.SGD(model.parameters(), lr=1e-2, momentum=0.9)

    optim.step() #gradient descent

print(model)

#print(f'Model after learning:**************:', model)
for key, value in model.named_parameters():
    print(f'model parameters key:*********:\n {key} \n')
    print(f'model parameters value:*********:\n {value} \n')

