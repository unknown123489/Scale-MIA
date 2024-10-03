# In this file, we demonstrate how the attacker estimates the essential parameters to launch the attack
# This corresponds to attack step 3 to 4
# For convenience, we have stored all the execution results in the data folder
# The user can directly execute the attack main file for evaluation

from torch.optim import SGD
import torch
import torchvision
from nets.CNNencoder import Encoder
from torchvision import transforms
import numpy as np
from torchvision.datasets import mnist
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import ToTensor
import scipy
#import scipy.stats
from scipy.stats import norm
from scipy.stats import laplace
import matplotlib.pyplot as plt
from datasets import load_dataset

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
batch_size = 100

toTensorTransform = transforms.Compose([transforms.ToTensor(),])
tiny_imagenet_train = load_dataset('Maysee/tiny-imagenet', split='train')
tiny_imagenet_val = load_dataset('Maysee/tiny-imagenet', split='valid')
class CustomDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.hf_dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        example = self.hf_dataset[idx]
        image = example['image']
        label = example['label']
        

        if self.transform:
            image = self.transform(image)
        if image.shape[0] == 1:
            return None
        else:
            return image, label
custom_val_dataset = CustomDataset(tiny_imagenet_val, transform=toTensorTransform)
custom_val_dataset = [item for item in custom_val_dataset if item is not None]
test_loader = DataLoader(custom_val_dataset, batch_size=batch_size, shuffle=True)
custom_train_dataset = CustomDataset(tiny_imagenet_train, transform=toTensorTransform)
custom_train_dataset = [item for item in custom_train_dataset if item is not None]
train_loader = DataLoader(custom_train_dataset, batch_size=batch_size, shuffle=True)

bright_small=torch.zeros(10000)
latent_small=torch.zeros(10000)

trained_model=Encoder()
trained_model.load_state_dict(torch.load("models/encoder_model_10_new_3.pkl"))
trained_model.eval()
trained_model.to(device)
trained_model.relu3.register_forward_hook(get_activation('relu3'))

for idx, (train_x, train_label) in enumerate(train_loader):
    if idx<int(10000/batch_size):
        train_x = train_x.to(device)
        train_label = train_label.to(device)
        output = trained_model(train_x)
        latent_batch=torch.mean(activation['relu3'].view(100,-1), dim=1).cpu()
        latent_small[idx * 100:(idx + 1) * 100] = latent_batch[0:100]

x = np.linspace(0.2, 1.7, 300)

# Estimate the distribution and obtain the critical h vector for the linear leakage primitive
norm_param_small = norm.fit(latent_small)
rv_norm_small = norm(norm_param_small[0], norm_param_small[1])
small_bins=rv_norm_small.ppf(np.linspace(0.001, 0.999, 2048))

torch.save(small_bins, "data/latent_small_bins_10_3.pt")
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
plt.savefig("figs/Stats-Tiny.pdf")
plt.show()
    



