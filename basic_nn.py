# Hello Machine Learning...!

# imports
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
import torchvision.transforms as transforms

# Create fully connected network
class NN(nn.Module):
    def __init__(self, input_size, num_classes): # 28x28 = 784
        super(NN, self).__init__()
        self.fc1 = nn.Linear(input_size, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Test if the model works with random data
# model = NN(784,10)
# x = torch.randn(64, 784)
# print(model(x).shape)

# Set Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
input_size = 784
num_classes = 10
learning_rate = 0.001
batch_size = 64
num_epochs = 5


# Load Data (root is folder, train is whether this is our train data, and transform is to transform to tensor to run in PyTorch)
train_dataset = datasets.MNIST(root = "dataset/", train = True, transform = transforms.ToTensor(), download = True) # Download if not already
train_loader = DataLoader(dataset = train_dataset, batch_size = batch_size, shuffle = True) # Shuffle is set to true
test_dataset = datasets.MNIST(root = "dataset/", train = False, transform = transforms.ToTensor(), download = True)
test_loader = DataLoader(dataset = test_dataset, batch_size = batch_size, shuffle = True)

# Initialize Network
model = NN(input_size = input_size, num_classes = num_classes).to(device)

# loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = learning_rate)

# Train the network
# One epoch means the model has seen all the images once
for epoch in range(num_epochs):
    print(f"Epoch {epoch}...")
    for batch_idx, (data, targets) in enumerate(train_loader):
        # Get data to cuda if possible
        data = data.to(device) # Move to device
        targets = targets.to(device) # move to device

        # Get data to correct shape
        # we want to make the tensor of the data into one dimension (unroll the matrix into a vector)
        data = data.reshape(data.shape[0], -1)

        #or
        # data = data.flatten()

        # Forward Pass
        scores = model(data)
        loss = criterion(scores, targets)
        
        # Backward Pass
        optimizer.zero_grad()
        loss.backward()

        # Gradient Descent or Adam Step
        optimizer.step()

# Check the accuracy on training and test to see how good our model is

def check_accuracy(loader, model):
    if loader.dataset.train:
        print("Checking accuracy on training data.")
    else:
        print("Checking accuracy on training data.")
    num_correct = 0
    num_samples = 0
    
    # Set model to evaluate mode.
    model.eval()

    # When we are checking for accuracy, we don't want to calculate any gradients (because this is unnecessary.)
    with torch.no_grad():
        for x,y in loader:
            x = x.to(device)
            y = y.to(device)
            x = x.reshape(x.shape[0], -1)

            scores = model(x)
            # 64 x 10
            # Get max of column of 10 (num of classes!)
            _, predictions = scores.max(1) # param is column
            num_correct += (predictions == y).sum()
            num_samples += predictions.size(0)

        print(f"Got {num_correct} / {num_samples} with accuracy: {float(num_correct)/float(num_samples)*100:.2f}%")

    # Return to training mode.
    model.train()
    return

check_accuracy(train_loader, model)
check_accuracy(test_loader, model)