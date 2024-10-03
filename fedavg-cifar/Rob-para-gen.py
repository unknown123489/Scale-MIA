from torch.optim import SGD
import torch
import torchvision
from torchvision import transforms
import numpy as np
from torchvision.datasets import mnist
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import scipy
#import scipy.stats
from scipy.stats import norm
from scipy.stats import laplace
import matplotlib.pyplot as plt

device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
batch_size = 100
transform = transforms.Compose([transforms.ToTensor(),])
train_dataset = torchvision.datasets.CIFAR10(root='./train', train=True, download=True, transform=transform)
test_dataset = torchvision.datasets.CIFAR10(root='./test', train=False, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size)

bright_small=torch.zeros(50000)
latent_small=torch.zeros(50000)

for idx, (train_x, train_label) in enumerate(train_loader):
    latent_batch=torch.mean(train_x.view(100,-1), dim=1).cpu()
    latent_small[idx * 100:(idx + 1) * 100] = latent_batch[0:100]

x = np.linspace(0.1, 4.0, 300)
norm_param_small = norm.fit(latent_small)
rv_norm_small = norm(norm_param_small[0], norm_param_small[1])
small_bins=rv_norm_small.ppf(np.linspace(0.001, 0.999, 32*8*8))

torch.save(small_bins, "data/latent_small_bins_rob_100.pt")
print(small_bins)

plt.figure()
plt.hist(latent_small.detach().cpu().numpy(), bins=32, density=True, label="small")
plt.plot(x, rv_norm_small.pdf(x), label="100%")
plt.legend()
plt.savefig("figs/Stats.png")
plt.show()
    



