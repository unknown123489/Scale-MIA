from torch.nn import Module
from torch import nn
import torch

#class Decoder(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.relu2 = nn.ReLU()
#        self.convtrans1=nn.ConvTranspose2d(256, 64, 3, stride=2, padding=2)
#        self.relu3 = nn.ReLU()
#        self.convtrans2=nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1)
#        self.relu4 = nn.ReLU()
#        self.convtrans3=nn.ConvTranspose2d(32, 16, 3, stride=2, padding=1)
#        self.relu5 = nn.ReLU()
#        self.convtrans4=nn.ConvTranspose2d(16, 3, 4, stride=2, padding=2)
#        self.sigmoid = nn.Sigmoid()
#
#    def forward(self, x):
#        y = self.convtrans1(x)
#        y = self.relu2(y)
#        y = self.convtrans2(y)
#        y = self.relu3(y)
#        y = self.convtrans3(y)
#        y = self.relu4(y)
#        y = self.convtrans4(y)
#        y = self.sigmoid(y)
#        return y

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.upsample1=nn.Upsample(scale_factor=2)
        self.relu2 = nn.ReLU()
        self.convtrans1=nn.ConvTranspose2d(256, 64, 4)
        self.relu3 = nn.ReLU()
        self.upsample2=nn.Upsample(scale_factor=2)
        self.relu4 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(num_features=64)
#        self.dropout1=torch.nn.Dropout(p=0.5, inplace=False)
        self.convtrans2 = nn.ConvTranspose2d(64, 32, 4)     
        self.relu5 = nn.ReLU()
        self.convtrans3=nn.ConvTranspose2d(32, 24, 4)
        self.relu6 = nn.ReLU()
        self.convtrans4=nn.ConvTranspose2d(24, 8, 3)
        self.relu7 = nn.ReLU()
        self.convtrans5=nn.ConvTranspose2d(8, 3, 3)
        self.relu=nn.ReLU()
#        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.upsample1(x)
        y = self.relu2(y)
        y = self.convtrans1(y)
        y = self.relu3(y)
        y = self.upsample2(y)
        y = self.relu4(y)
        y = self.batchnorm1(y)
#        y = self.dropout1(y)
        y = self.convtrans2(y)
        y = self.relu5(y)
        y = self.convtrans3(y)
        y = self.relu6(y)
        y = self.convtrans4(y)
        y = self.relu7(y)
        y = self.convtrans5(y)
        y = self.relu(y)
        return y





