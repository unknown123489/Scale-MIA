from torch.nn import Module
from torch import nn

class Decoder(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.convtrans1=nn.ConvTranspose2d(in_channels=64, out_channels=32,stride=2, padding=2, output_padding=1, kernel_size=5)
        self.relu3 = nn.ReLU()
        self.convtrans2=nn.ConvTranspose2d(in_channels=32, out_channels=8,stride=2, padding=2, kernel_size=5)
        self.relu4 = nn.ReLU()
        self.convtrans3=nn.ConvTranspose2d(in_channels=8, out_channels=1,stride=2, padding=2, kernel_size=4)
        
    def forward(self, x):
        y = self.convtrans1(x)
        y = self.relu3(y)
        y = self.convtrans2(y)
        y = self.relu4(y)
        y = self.convtrans3(y)
        
        return y
