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
from nets.CNNencoder import Encoder
from nets.CNNdecoder import Decoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)
batch_size = 50

train_dataset = mnist.FashionMNIST(root='./train', train=True, transform=ToTensor(), download=True)
test_dataset = mnist.FashionMNIST(root='./test', train=False, transform=ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)

encoder = Encoder().to(device)
decoder = Decoder().to(device)

params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

lr=0.0001
optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

loss_fn = MSELoss()
all_epoch = 200
prev_acc = 0
train_loss=[]

for current_epoch in range(all_epoch): 

    encoder.train()
    decoder.train()
    epoch_loss=0
    for idx, (train_x, train_label) in enumerate(train_loader):
        if idx<int(60000/batch_size):
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

torch.save(encoder.state_dict(), "models/encoder_model_100_new_2.pkl") 
torch.save(decoder.state_dict(), "models/decoder_model_100_new_2.pkl") 
    
encoder.eval()
decoder.eval()
   
           
for idx, (test_x, test_label) in enumerate(test_loader):
    if idx==0:
        test_x=test_x.to(device)
        encoded_data=encoder(test_x)
        decoded_data=decoder(encoded_data)
        input_data=test_x.detach().cpu().view(batch_size,-1).numpy()
        recovered=decoded_data.detach().cpu().view(batch_size,-1).numpy()
        np.save("data/input.npy",input_data)
        np.save("data/recovered.npy",recovered)
        
plt.figure()
plt.plot(range(all_epoch), train_loss)
plt.savefig("figs/loss_trend")
#plt.show()        
    
    
    
    
    
    
    
    

