import torch.nn as nn
from torch.nn import Module


cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'ML', 512, 512, 'ML'],
}


class VGG(nn.Module):
    def __init__(self, vgg_name='VGG11'):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name])
        self.fc1=nn.Linear(512*4*4, 1024)
        self.relu1=nn.ReLU()
        self.fc2=nn.Linear(1024, 200)
        self.relu2=nn.ReLU()
        self.fc3=nn.Linear(200, 10)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.relu1(out)
        out = self.fc2(out)
        out = self.relu2(out)
        out = self.fc3(out)
        return out

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            elif x == 'ML':
                layers += [nn.MaxPool2d(kernel_size=1, stride=1)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)
