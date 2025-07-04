import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self, num_classes=4):
        super(Net, self).__init__()
        # Convolutional layers + batchnorm
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)

        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)

        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm2d(64)

        self.pool = nn.MaxPool2d(2, 2)

        #
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 64, 64)
            x = self.pool(self.bn1(F.relu(self.conv1(dummy))))
            x = self.pool(self.bn2(F.relu(self.conv2(x))))
            x = self.pool(self.bn3(F.relu(self.conv3(x))))
            flat_size = x.view(1, -1).shape[1]

        # Fully connected layers
        self.fc1 = nn.Linear(flat_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.pool(self.bn1(F.relu(self.conv1(x))))
        x = self.pool(self.bn2(F.relu(self.conv2(x))))
        x = self.pool(self.bn3(F.relu(self.conv3(x))))
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
