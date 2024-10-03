from torch.nn import Module
from torch import nn
import torch

class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.relu2 = nn.ReLU()
        self.convtrans1=nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.convtrans2=nn.ConvTranspose2d(32, 12, 4, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.convtrans3=nn.ConvTranspose2d(12, 3, 4, stride=2, padding=1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        y = self.convtrans1(x)
        y = self.relu2(y)
        y = self.convtrans2(y)
        y = self.relu3(y)
        y = self.convtrans3(y)
        y = self.sigmoid(y)
        return y
