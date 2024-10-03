import torch.nn as nn
from torch.nn import Module
import torch

class Decoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.convtrans1=nn.ConvTranspose2d(256, 128, 3, padding=1)
        self.relu1 = nn.ReLU()
        self.convtrans9=nn.ConvTranspose2d(128, 128, 3, padding=1)
        self.relu9 = nn.ReLU()
        self.batchnorm1 = nn.BatchNorm2d(num_features=128)
        
        self.upsample2=nn.Upsample(scale_factor=2)
        self.convtrans2=nn.ConvTranspose2d(128, 64, 3, padding=1)
        self.relu2 = nn.ReLU()
        self.convtrans8=nn.ConvTranspose2d(64, 64, 3, padding=1)
        self.relu8 = nn.ReLU()
        self.batchnorm2 = nn.BatchNorm2d(num_features=64)

        self.upsample3=nn.Upsample(scale_factor=2)
        self.convtrans3=nn.ConvTranspose2d(64, 32, 3, padding=1)
        self.relu3 = nn.ReLU()
        self.convtrans7=nn.ConvTranspose2d(32, 32, 3, padding=1)
        self.relu7 = nn.ReLU()
        self.batchnorm3 = nn.BatchNorm2d(num_features=32)
        
        self.upsample4=nn.Upsample(scale_factor=2)
        self.convtrans4=nn.ConvTranspose2d(32, 16, 3, padding=1)
        self.relu4 = nn.ReLU()
        self.convtrans5=nn.ConvTranspose2d(16, 3, 3, padding=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = self.convtrans1(x)
        y = self.relu1(y)
        y = self.convtrans9(y)
        y = self.relu9(y)
        y = self.batchnorm1(y)
        
        y = self.upsample2(y)
        y = self.convtrans2(y)
        y = self.relu2(y)
        y = self.convtrans8(y)
        y = self.relu8(y)
        y = self.batchnorm2(y)
        
        y = self.upsample3(y)
        y = self.convtrans3(y)
        y = self.relu3(y)
        y = self.convtrans7(y)
        y = self.relu7(y)
        y = self.batchnorm3(y)
        
        y = self.upsample4(y)
        y = self.convtrans4(y)
        y = self.relu4(y)
        y = self.convtrans5(y)
        y = self.sigmoid(y)
        
        return y



#class Decoder(nn.Module):
#    def __init__(self):
#        super().__init__()
#        self.relu8 = nn.ReLU()
#        self.convtrans6=nn.ConvTranspose2d(512, 256, 4)
#        self.upsample1=nn.Upsample(scale_factor=2)
#        self.relu2 = nn.ReLU()
#        self.convtrans1=nn.ConvTranspose2d(256, 64, 4)
#        self.relu3 = nn.ReLU()
#        self.upsample2=nn.Upsample(scale_factor=2)
#        self.relu4 = nn.ReLU()
#        self.batchnorm1 = nn.BatchNorm2d(num_features=64)
#        self.dropout1=nn.Dropout(p=0.5, inplace=False)
#        self.convtrans2 = nn.ConvTranspose2d(64, 32, 4)     
#        self.relu5 = nn.ReLU()
#        self.convtrans3=nn.ConvTranspose2d(32, 24, 4)
#        self.relu6 = nn.ReLU()
#        self.convtrans4=nn.ConvTranspose2d(24, 8, 3)
#        self.relu7 = nn.ReLU()
#        self.convtrans5=nn.ConvTranspose2d(8, 3, 3)
#        self.sigmoid = nn.Sigmoid()

#    def forward(self, x):
#        y = self.convtrans6(x)
#        y = self.relu8(y)
#        y = self.upsample1(y)
#        y = self.relu2(y)
#        y = self.convtrans1(y)
#        y = self.relu3(y)
#        y = self.upsample2(y)
#        y = self.relu4(y)
#        y = self.batchnorm1(y)
#        y = self.dropout1(y)
#        y = self.convtrans2(y)
#        y = self.relu5(y)
#        y = self.convtrans3(y)
#        y = self.relu6(y)
#        y = self.convtrans4(y)
#        y = self.relu7(y)
#        y = self.convtrans5(y)
#        y = self.sigmoid(y)
#        return y