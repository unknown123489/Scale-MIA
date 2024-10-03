# In this file, we demonstrate how the attacker estimates the essential parameters to launch the attack
# This corresponds to attack step 3 to 4
# For convenience, we have stored all the execution results in the data folder
# The user can directly execute the attack main file for evaluation

from torch.optim import SGD
import torch
from nets.CNNencoder import Encoder
import numpy as np
from torchvision.transforms import ToTensor
import scipy
#import scipy.stats
from scipy.stats import norm
from scipy.stats import laplace
import matplotlib.pyplot as plt
import pandas as pd

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
batch_size = 100
dataset=pd.read_csv("hmnist_28_28_L.csv")
X=torch.Tensor(dataset.iloc[:,:-1].values).view(-1,1,28,28)/255
Y=torch.Tensor(dataset.iloc[:,-1:].values).view(5000,1)

trainset_x=torch.zeros(3000,1,28,28)
trainset_x[0:1200,:,:,:]=X[0:1200,:,:,:]
trainset_x[1200:3000,:,:,:]=X[3200:5000,:,:,:]
#testset_x=X[4000:,:,:,:]

bright_small=torch.zeros(3000)
latent_small=torch.zeros(3000)


trained_model=Encoder()
trained_model.load_state_dict(torch.load("models/encoder_model_100_new.pkl"))
trained_model.eval()
trained_model.to(device)
#trained_model.relu3.register_forward_hook(get_activation('relu3')) have this


for idx in range(int(3000/batch_size)):
    train_x = trainset_x[idx*batch_size:(idx+1)*batch_size,:,:,:].to(device)
    output = trained_model(train_x)
    latent_batch=torch.mean(output.view(100,-1), dim=1).detach().cpu()
    latent_small[idx * 100:(idx + 1) * 100] = latent_batch[0:100]


x = np.linspace(0.2, 1.0, 500)
# Estimate the distribution and obtain the critical h vector for the linear leakage primitive
norm_param_small = norm.fit(latent_small)
rv_norm_small = norm(norm_param_small[0], norm_param_small[1])
small_bins=rv_norm_small.ppf(np.linspace(0.001, 0.999, 1024))

torch.save(small_bins, "data/CNN_latent_small_bins_100_new.pt")

print(small_bins)

# We also plot the LSR distribution of the dataset
# The figure can be founs in the figs folder
plt.figure(figsize=(5.9,5.3))
plt.hist(latent_small.detach().cpu().numpy(), bins=48, density=True, label="LSR Dist", color='g')
plt.plot(x, rv_norm_small.pdf(x), label="Estimated Dist", color='r')
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(range(0,5,1),["0","1.0","2.0","3.0","4.0"],fontsize=16)
plt.xlabel("LSR Brightness",fontsize=16)
plt.ylabel("Density",fontsize=16)
plt.savefig("figs/Stats-HMNIST.pdf")
plt.show()
    











