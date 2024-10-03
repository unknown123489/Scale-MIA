import torch.nn as nn
from torch.nn import Module
import torch


class Decoder(nn.Module):

    def __init__(self):
        super().__init__()
        self.relu2 = nn.ReLU()
        self.convtrans1=nn.ConvTranspose2d(32, 32, 4, stride=2, padding=1)
        self.relu3 = nn.ReLU()
        self.convtrans2=nn.ConvTranspose2d(32, 24, 4, stride=2, padding=1)
        self.relu4 = nn.ReLU()
        self.convtrans3=nn.ConvTranspose2d(24, 3, 4, stride=2, padding=1)
        self.relu=nn.ReLU()
#        self.sigmoid = nn.Sigmoid()


    def forward(self, x):
        x = x.view(-1, 32, 4, 4)
        y = self.convtrans1(x)
        y = self.relu2(y)
        y = self.convtrans2(y)
        y = self.relu3(y)
        y = self.convtrans3(y)
        y=self.relu(y)
#        y = self.sigmoid(y)
        return y


#Class Decoder(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.convtrans1 = nn.ConvTranspose2d( 256, 128, 4, padding=1)
#        self.relu1 = nn.ReLU()
#        self.convtrans9 = nn.ConvTranspose2d(128, 96, 4, padding=1)
#        self.relu9 = nn.ReLU()
#        self.batchnorm1 = nn.BatchNorm2d(num_features=96)
#
#        self.upsample2 = nn.Upsample(scale_factor=2)
#        self.convtrans2 = nn.ConvTranspose2d(96, 64, 3, padding=1)
#        self.relu2 = nn.ReLU()
#        self.convtrans8 = nn.ConvTranspose2d(64, 64, 3, padding=1)
#        self.relu8 = nn.ReLU()
#        self.batchnorm2 = nn.BatchNorm2d(num_features=64)
#
#        self.upsample3 = nn.Upsample(scale_factor=2)
#        self.convtrans3 = nn.ConvTranspose2d(64, 32, 3, padding=1)
#        self.relu3 = nn.ReLU()
#        self.convtrans7 = nn.ConvTranspose2d(32, 32, 3, padding=1)
#        self.relu7 = nn.ReLU()
#        self.batchnorm3 = nn.BatchNorm2d(num_features=32)
#
#        self.upsample4 = nn.Upsample(scale_factor=2)
#        self.convtrans4 = nn.ConvTranspose2d(32, 16, 3, padding=1)
#        self.relu4 = nn.ReLU()
#        self.convtrans5 = nn.ConvTranspose2d(16, 3, 3, padding=1)
#        self.sigmoid = nn.Sigmoid()
#
#    def forward(self, x):
#        x = x.view(-1, 256, 2, 2)
#        y = self.convtrans1(x)
#        y = self.relu1(y)
#        y = self.convtrans9(y)
#        y = self.relu9(y)
#        y = self.batchnorm1(y)
#
#        y = self.upsample2(y)
#        y = self.convtrans2(y)
#        y = self.relu2(y)
#        y = self.convtrans8(y)
#        y = self.relu8(y)
#        y = self.batchnorm2(y)
#
#        y = self.upsample3(y)
#        y = self.convtrans3(y)
#        y = self.relu3(y)
#        y = self.convtrans7(y)
#        y = self.relu7(y)
#        y = self.batchnorm3(y)
#
#        y = self.upsample4(y)
#        y = self.convtrans4(y)
#        y = self.relu4(y)
#        y = self.convtrans5(y)
#        y = self.sigmoid(y)
#
#        return y