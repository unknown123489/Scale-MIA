from torch.nn import Module
from torch import nn


class Encoder(nn.Module):
    
    def __init__(self):
        super().__init__()      
        self.conv1 = nn.Conv2d(3, 16, 4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, stride=2, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(64, 128, 4, stride=2, padding=1)
        self.relu1 = nn.ReLU()
        self.relu2 = nn.ReLU()
        self.relu3 = nn.ReLU()
        self.relu4 = nn.ReLU()
        
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.relu3(self.conv3(x))
        x = self.conv4(x)
        x = self.relu4(x)        
        return x
