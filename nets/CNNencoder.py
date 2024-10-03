from torch.nn import Module
from torch import nn


class Encoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        ### Convolutional section
#        self.conv1 = nn.Conv2d(1, 16, 5)
#        self.relu1 = nn.ReLU()
#        self.pool1 = nn.MaxPool2d(2)
#        self.conv2 = nn.Conv2d(16, 64, 5)
#        self.relu2 = nn.ReLU()
#        self.pool2 = nn.MaxPool2d(2)
        
        self.conv1 = nn.Conv2d(3, 12, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(12, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        
    def forward(self, x):
#        y = self.conv1(x)
#        y = self.relu1(y)
#        y = self.pool1(y)
#        y = self.conv2(y)
#        y = self.relu2(y)
#        y = self.pool2(y)
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.conv3(x)
        x = self.relu3(x)        
        return x
