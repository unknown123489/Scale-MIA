# This file is the training process of the surrogate autoencoder, corresponds to attack steps 1 to 3
# We have plotted the training loss in the figs/loss_trend figure
# For convenience, we have already conducted the training process and stored the already trained parameters in the models folder
# The user can directly execute the attack main file for evaluation

import matplotlib.pyplot as plt 
import numpy as np 
import random 
import torch
import torchvision
from torchvision.transforms import ToTensor
from torch import nn
from torch.nn import MSELoss
import torch.nn.functional as F
import torch.optim as optim
from nets.CNNencoder import Encoder
from nets.CNNdecoder import Decoder
import pandas as pd

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print(device)
batch_size = 20

dataset=pd.read_csv("hmnist_28_28_L.csv")
X=torch.Tensor(dataset.iloc[:,:-1].values).view(-1,1,28,28)/255
Y=torch.Tensor(dataset.iloc[:,-1:].values).view(5000,1)

index=torch.randint(0,5000,(4000,))

trainset_x=X[index,:,:,:] #0:4000
testset_x=X[4000:,:,:,:]

encoder = Encoder().to(device)
decoder = Decoder().to(device)

params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

lr=0.0001
optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

loss_fn = MSELoss()
all_epoch = 1500
prev_acc = 0
train_loss=[]

for current_epoch in range(all_epoch): 

    encoder.train()
    decoder.train()
    epoch_loss=0
    for idx in range(int(4000/batch_size)):
        if idx<int(4000/batch_size):
#            print(train_x.shape)
            train_x=trainset_x[idx*batch_size:(idx+1)*batch_size,:,:,:].to(device)
            encoded_data=encoder(train_x)
#            print(encoded_data.shape)
            decoded_data=decoder(encoded_data)
            loss = loss_fn(decoded_data, train_x)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss=epoch_loss+loss.item()/200
        else:
            break
    train_loss.append(epoch_loss)
    if current_epoch%10==0:
        print(epoch_loss)

torch.save(encoder.state_dict(), "models/encoder_model_100_new_2.pkl") 
torch.save(decoder.state_dict(), "models/decoder_model_100_new_2.pkl") 
    
encoder.eval()
decoder.eval()
  
           
for idx in range(int(1000/batch_size)):
    if idx==0:
        test_x=testset_x[idx*batch_size:(idx+1)*batch_size,:,:,:].to(device)
        encoded_data=encoder(test_x)
        decoded_data=decoder(encoded_data)
        input_data=test_x.detach().cpu().view(batch_size,-1).numpy()
        recovered=decoded_data.detach().cpu().view(batch_size,-1).numpy()
        np.save("data/cnnnet_input.npy",input_data)
        np.save("data/cnnnet_recovered.npy",recovered)
        
plt.figure()
plt.plot(range(all_epoch), train_loss)
plt.savefig("figs/loss_trend")
#plt.show()        
    
    
    
    
    
    
    
    

