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
        
        self.conv1 = nn.Conv2d(3, 12, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(12, 32, 3, stride=1, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, stride=1, padding=1)
        self.relu1 = nn.ReLU()
        self.avg1 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu2 = nn.ReLU()
        self.avg2 = nn.AvgPool2d(kernel_size=2, stride=2)
        self.relu3 = nn.ReLU()
        self.avg3 = nn.AvgPool2d(kernel_size=2, stride=2)
        
    def forward(self, x):
#        y = self.conv1(x)
#        y = self.relu1(y)
#        y = self.pool1(y)
#        y = self.conv2(y)
#        y = self.relu2(y)
#        y = self.pool2(y)
        x = self.avg1(self.relu1(self.conv1(x)))
        x = self.avg2(self.relu2(self.conv2(x)))
        x = self.avg3(self.relu3(self.conv3(x)))     
        return x
