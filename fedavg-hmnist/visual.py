# This file is used to plot the original and reconstructed samples
# For convenience, we have plotted all figures and they can be found in the figs folder

import matplotlib.pyplot as plt 
import numpy as np     
import torch   
from torch.nn import MSELoss      
mse_fn= MSELoss()   
        
batch_size=64
dim=28
channel=1       
input_data=torch.load("data/data_batch_64.pt")
recovered=torch.load("data/recoved_64.pt")
sorted_list=torch.load("data/list_64.pt")
input_data=input_data.reshape(batch_size,channel,dim,dim)
recovered=recovered.reshape(batch_size,channel,dim,dim)
   
plt.figure()
img=torch.zeros((dim,dim,channel))
for i in range(batch_size):
    for j in range(channel):
        img[:,:,j]=input_data[sorted_list[batch_size-1-i],j,:,:]
    plt.subplot(8,int(batch_size/8),i+1)
    plt.imshow(img.detach().numpy(), cmap="gray")
    plt.grid(False)
    plt.axis('off')
plt.savefig("figs/original_sample")

plt.figure()

img=torch.zeros((dim,dim,channel))
for i in range(batch_size):
    for j in range(channel):
        img[:,:,j]=recovered[i,j,:,:]
    plt.subplot(8,int(batch_size/8),i+1)
    plt.imshow(img.detach().numpy(), cmap="gray")
    plt.grid(False)
    plt.axis('off')
plt.savefig("figs/reconstructed_sample")
    
    
    
    
    
    