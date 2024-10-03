from torch.nn import Module
from torch import nn


class Classifier(Module):
    def __init__(self):
        super(Classifier, self).__init__()
        self.ln1 = nn.Linear(32*32*3, 32*8*8) #32-->16
        self.ln2 = nn.Linear(32*8*8, 32*8*8) #16-->8
        self.conv3 = nn.Conv2d(32, 64, 4, stride=2, padding=1)  #8-->4
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
        x = x.view(x.shape[0], -1)
        x = self.relu1(self.ln1(x))
        x = self.relu2(self.ln2(x))
        x = x.view(x.shape[0],32,8,8)
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

