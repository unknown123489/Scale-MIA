# In this file, we demonstrate how the attacker estimates the essential parameters to launch the attack
# This corresponds to attack step 3 to 4
# For convenience, we have stored all the execution results in the data folder
# The user can directly execute the attack main file for evaluation

from torch.optim import SGD
import torch
from nets.CNNencoder import Encoder
import numpy as np
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import scipy
#import scipy.stats
from scipy.stats import norm
from scipy.stats import laplace
import matplotlib.pyplot as plt

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
batch_size = 100
train_dataset = mnist.FashionMNIST(root='./train', train=True, transform=ToTensor(), download=True)
test_dataset = mnist.FashionMNIST(root='./test', train=False, transform=ToTensor(), download=True)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

bright_small=torch.zeros(60000)
latent_small=torch.zeros(60000)


trained_model=Encoder()
trained_model.load_state_dict(torch.load("models/encoder_model_100_new_2.pkl"))
trained_model.eval()
trained_model.to(device)
trained_model.relu3.register_forward_hook(get_activation('relu3'))


for idx, (train_x, train_label) in enumerate(train_loader):
    if idx<int(60000/batch_size):
        train_x = train_x.to(device)
        train_label = train_label.to(device)
        output = trained_model(train_x)
        latent_batch=torch.mean(activation['relu3'].view(100,-1), dim=1).cpu()
        latent_small[idx * 100:(idx + 1) * 100] = latent_batch[0:100]
    else:
        break

x = np.linspace(0.0, 1.4, 300)

# Estimate the distribution and obtain the critical h vector for the linear leakage primitive
norm_param_small = norm.fit(latent_small)
rv_norm_small = norm(norm_param_small[0], norm_param_small[1])
small_bins=rv_norm_small.ppf(np.linspace(0.001, 0.999, 1024))

torch.save(small_bins, "data/latent_small_bins_100_new_2.pt")

print(small_bins)

# We also plot the LSR distribution of the dataset
# The figure can be founs in the figs folder
plt.figure(figsize=(5.9,5.3))
plt.hist(latent_small.detach().cpu().numpy(), bins=48, density=True, label="LSR Dist", color='g')
plt.plot(x, rv_norm_small.pdf(x), label="Estimated Dist", color='r')
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("LSR Brightness",fontsize=16)
plt.ylabel("Density",fontsize=16)
plt.savefig("figs/Stats-FMNIST.pdf")
plt.show()
    


