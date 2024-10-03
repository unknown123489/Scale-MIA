from torch.nn import Module
from torch import nn


class Classifier(Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 4, stride=2, padding=1) #64-->32
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)  #32-->16
        self.conv3 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  #16-->8
        self.conv4 = nn.Conv2d(64, 128, 4, stride=2, padding=1)  #8-->4
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        self.fc1 = nn.Linear(128*4*4, 2048)
        self.relu5 = nn.ReLU()
        self.fc2 = nn.Linear(2048, 1024)
        self.relu6 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 200)
        self.relu7 = nn.ReLU()

    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.conv4(x)
        y = self.relu4(x)
        y = y.view(y.shape[0], -1)
        y = self.fc1(y)
        y = self.relu5(y)
        y = self.fc2(y)
        y = self.relu6(y)
        y = self.fc3(y)
        y = self.relu7(y)
        
        return y

