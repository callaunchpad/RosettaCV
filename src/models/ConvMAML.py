import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Create 3 layers of CNN
        # Flatten
        # Use two layers for fully connected layers
        self.cnn1 = torch.nn.Conv2d(1, 28, kernel_size=3, padding=1)
        self.activation1 = F.relu
        self.cnn2 = torch.nn.Conv2d(28, 28, kernel_size=3, padding=1)
        self.activation2 = F.relu
        self.pool2 = torch.nn.MaxPool2d(2)
        self.cnn3 = torch.nn.Conv2d(28,28, kernel_size=3, padding=1)
        self.activation3 = F.relu
        self.flatten = nn.Flatten()
        self.linear1 = torch.nn.Linear(10976 // 2, 2000)
        self.linear2 = torch.nn.Linear(2000, 1000)
		self.softmax = torch.nn.SoftMax()

    def forward(self, x):
        # The hidden activation should be relu but no output activation
        x = self.cnn1(x)
        x = self.activation1(x)
        x = self.cnn2(x)
        x = self.activation2(x)
        x = self.pool2(x)
        x = self.cnn3(x)
        x = self.activation3(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.linear2(x)
		x = self.softmax(x)
        return x
