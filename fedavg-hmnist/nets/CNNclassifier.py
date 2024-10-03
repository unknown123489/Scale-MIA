from torch.nn import Module
from torch import nn


class Classifier(Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 12, 4, stride=2, padding=2)
        self.conv2 = nn.Conv2d(12, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.fc1 = nn.Linear(64*4*4, 1024)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(512, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        y = self.relu3(x)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        
        return y

