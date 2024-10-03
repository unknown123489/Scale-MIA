from torch.nn import Module
from torch import nn


class Classifier(Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 12, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(12, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.avg1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()
        self.avg2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu3 = nn.ReLU()
        self.avg3 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.fc1 = nn.Linear(64*4*4, 1024)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(1024, 512)
        self.relu4 = nn.ReLU()
        self.fc3 = nn.Linear(512, 10)
        self.relu5 = nn.ReLU()

    def forward(self, x):
        x = self.avg1(self.relu1(self.conv1(x)))
        x = self.avg2(self.relu2(self.conv2(x)))
        y = self.avg3(self.relu3(self.conv3(x)))
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu3(y)
        y = self.fc2(y)
        y = self.relu4(y)
        y = self.fc3(y)
        y = self.relu5(y)
        
        return y

