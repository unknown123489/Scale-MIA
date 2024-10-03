# This file is the training process of the surrogate autoencoder, corresponds to attack steps 1 to 3
# We have plotted the training loss in the figs/loss_trend figure
# For convenience, we have already conducted the training process and stored the already trained parameters in the models folder
# The user can directly execute the attack main file for evaluation

import matplotlib.pyplot as plt 
import numpy as np 
import random 
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch import nn
from torch.nn import MSELoss
import torch.nn.functional as F
import torch.optim as optim
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", default="CNN")
args = parser.parse_args()

name=args.model_name

if name=="CNN":
    from nets.CNNencoder import Encoder
    from nets.CNNdecoder import Decoder
elif name=="Alexnet":
    from nets.Alexnetencoder import Encoder
    from nets.Alexnetdecoder import Decoder
elif name=="Resnet":
    from nets.Resnetencoder import *
    from nets.Resnetdecoder import *
elif name=="Vit":
    from nets.Vitencoder import *
    from nets.Vitdecoder import *
elif name=="Vggnet":
    from nets.Vggnetencoder import Encoder
    from nets.Vggnetdecoder import Decoder
    

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print(device)

# The sample_num parameter controls the size of the auxiliary dataset
# The user can change it to different numbers for different auxiliary dataset sizes

sample_num=10000
batch_size = 100
transform = transforms.Compose([transforms.ToTensor(),])
train_dataset = torchvision.datasets.CIFAR10(root='./train', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./test', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

if name=="Vit":
    encoder = Encoder(
        image_size=32,
        patch_size=8,
        num_classes=10,
        channels=3,
        dim=512,
        depth=2,
        heads=2,
        mlp_dim=512,
        dropout=0,
        emb_dropout=0
        ).to(device)
        
elif name=="Resnet":
    encoder=Encoder(ResidualBlock, [2, 2, 2]).to(device)
else:
    encoder = Encoder().to(device)
decoder = Decoder().to(device)

params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

lr=0.0001
optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

loss_fn = MSELoss()
all_epoch = 800
prev_acc = 0
train_loss=[]

for current_epoch in range(all_epoch): 
    encoder.train()
    decoder.train()
    epoch_loss=0
    for idx, (train_x, train_label) in enumerate(train_loader):
        if idx<int(sample_num/batch_size):
#            print(train_x.shape)
            train_x=train_x.to(device)
            encoded_data=encoder(train_x)
#            print(encoded_data.shape)
            decoded_data=decoder(encoded_data)
            loss = loss_fn(decoded_data, train_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss=epoch_loss+loss.item()/300
        else:
            break
    train_loss.append(epoch_loss)
    print(epoch_loss)

torch.save(encoder.state_dict(), "models/encoder_model_"+name+".pkl") 
torch.save(decoder.state_dict(), "models/decoder_model_"+name+".pkl") 
    
encoder.eval()
decoder.eval()

plt.figure()
plt.plot(range(all_epoch), train_loss)
plt.savefig("figs/loss_trend")
plt.show()        
    
    
    
    
    
    
    
    

