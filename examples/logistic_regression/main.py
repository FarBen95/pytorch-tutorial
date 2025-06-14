from random import shuffle
import torch
import torch.nn as nn
import torch.utils
import torchvision
import torchvision.transforms as transforms


# Hyper-parameters 
input_size = 28 * 28
num_classes = 10
learning_rate = 0.01
batch_size = 100
num_epochs = 10

# MNIST dataset (images and labels)
train_dataset = torchvision.datasets.MNIST("../../data",
                                           train = True,
                                           transform=transforms.ToTensor(),
                                           download= True)

test_dataset = torchvision.datasets.MNIST(root="../../data",
                                          train=False,
                                          transform=transforms.ToTensor())

# Data loader (input pipeline)
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=True)

# Logistic regression model
model = nn.Linear(input_size, num_classes)

# Loss and optimizer
# nn.CrossEntropyLoss() computes softmax internally
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# Train the model
total_iter = len(train_loader)
for epoch in range(num_epochs):
    for i, (inputs, labels) in enumerate(train_loader):
        inputs = inputs.reshape(-1, input_size)
        
        preds = model(inputs)
        
        loss = criterion(preds, labels)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if (i+1) % 100 == 0:
            print("epoch [{}/{}], iteration [{}/{}], loss: {:.4f}"
                  .format(epoch+1, num_epochs, i+1, total_iter, loss.item()))

# Test the model
# In test phase, we don't need to compute gradients (for memory efficiency)
with torch.no_grad():
    correct = 0
    total = 0
    for inputs, labels in test_loader:
        inputs = inputs.reshape(-1, input_size)
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (labels==preds).sum()
        
    print("Accuracy: {:.4f} %".format(100*correct/total))

# Save the model checkpoint
torch.save(model.state_dict(), 'model.ckpt')