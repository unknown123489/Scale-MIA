# In this file, we demonstrate how the attacker estimates the essential parameters to launch the attack
# This corresponds to attack step 3 to 4
# For convenience, we have stored all the execution results in the data folder
# The user can directly execute the attack main file for evaluation
# This file is for the targeted attack. 
# In this example, we assume the attacker collects the ship, dog, and frog classes out of the 10 classes in the CIFAR-10 dataset

from torch.optim import SGD
import torch
import torchvision
from nets.CNNencoder import Encoder
from torchvision import transforms
import numpy as np
from torchvision.datasets import mnist
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToTensor
import scipy
#import scipy.stats
from scipy.stats import norm
from scipy.stats import laplace
import matplotlib.pyplot as plt
from torchvision.datasets import CIFAR10

activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook
sample_num=15000
device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
batch_size = 100
transform = transforms.Compose([transforms.ToTensor(),])

trainset = CIFAR10(root='./train', train=True, download=True)
# , transform = transform_no_aug)
testset = CIFAR10(root='./test', train=False, download=True)
classDict = {'plane': 0, 'car': 1, 'bird': 2, 'cat': 3, 'deer': 4,
             'dog': 5, 'frog': 6, 'horse': 7, 'ship': 8, 'truck': 9}

# Separating trainset/testset data/label
x_train = trainset.data
x_test = testset.data
y_train = trainset.targets
y_test = testset.targets

# Define a function to separate CIFAR classes by class index

def get_class_i(x, y, i):
    # Convert to a numpy array
    y = np.array(y)
    # Locate position of labels that equal to i
    pos_i = np.argwhere(y == i)
    # Convert the result into a 1-D list
    pos_i = list(pos_i[:, 0])
    # Collect all data that match the desired label
    x_i = [x[j] for j in pos_i]

    return x_i

class DatasetMaker(Dataset):
    def __init__(self, datasets, transformFunc=transform):
        """
        datasets: a list of get_class_i outputs, i.e. a list of list of images for selected classes
        """
        self.datasets = datasets
        self.lengths = [len(d) for d in self.datasets]
        self.transformFunc = transformFunc

    def __getitem__(self, i):
        class_label, index_wrt_class = self.index_of_which_bin(self.lengths, i)
        img = self.datasets[class_label][index_wrt_class]
        img = self.transformFunc(img)
        return img, class_label

    def __len__(self):
        return sum(self.lengths)

    def index_of_which_bin(self, bin_sizes, absolute_index, verbose=False):
        """
        Given the absolute index, returns which bin it falls in and which element of that bin it corresponds to.
        """
        # Which class/bin does i fall into?
        accum = np.add.accumulate(bin_sizes)
        if verbose:
            print("accum =", accum)
        bin_index = len(np.argwhere(accum <= absolute_index))
        if verbose:
            print("class_label =", bin_index)
        # Which element of the fallent class/bin does i correspond to?
        index_wrt_class = absolute_index - np.insert(accum, 0, 0)[bin_index]
        if verbose:
            print("index_wrt_class =", index_wrt_class)

        return bin_index, index_wrt_class
cat_dog_trainset = \
    DatasetMaker(
        [get_class_i(x_train, y_train, classDict['ship']),
         get_class_i(x_train, y_train, classDict['dog']),
         get_class_i(x_train, y_train, classDict['frog'])
         ],
        transform
    )
cat_dog_testset = \
    DatasetMaker(
        [get_class_i(x_test, y_test, classDict['ship']),
         get_class_i(x_test, y_test, classDict['dog']),
         get_class_i(x_train, y_train, classDict['frog'])
         ],
         transform
    )

# Create datasetLoaders from trainset and testset
train_loader = DataLoader(
    cat_dog_trainset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(
    cat_dog_testset, batch_size=batch_size, shuffle=True)

bright_small=torch.zeros(sample_num)
latent_small=torch.zeros(sample_num)

trained_model=Encoder()
trained_model.load_state_dict(torch.load("models/encoder_model_targeted_new.pkl"))
trained_model.eval()
trained_model.to(device)
trained_model.relu3.register_forward_hook(get_activation('relu3'))


for idx, (train_x, train_label) in enumerate(train_loader):
    if idx<int(sample_num/batch_size):
        train_x = train_x.to(device)
        train_label = train_label.to(device)
        output = trained_model(train_x)
        latent_batch=torch.mean(activation['relu3'].view(100,-1), dim=1).cpu()
        latent_small[idx * 100:(idx + 1) * 100] = latent_batch[0:100]
    else:
        break

x = np.linspace(1.6, 5.6, 800)

norm_param_small = norm.fit(latent_small)
rv_norm_small = norm(norm_param_small[0], norm_param_small[1])
small_bins=rv_norm_small.ppf(np.linspace(0.001, 0.999, 1024))

torch.save(small_bins, "data/latent_small_bins_targeted_new.pt")
print(small_bins)

plt.figure(figsize=(5.9,5.3))
plt.hist(latent_small.detach().cpu().numpy(), bins=48, density=True, label="LSR Dist", color='g')
plt.plot(x, rv_norm_small.pdf(x), label="Estimated Dist", color='r')
plt.legend(fontsize=14)
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.xlabel("LSR Brightness",fontsize=16)
plt.ylabel("Density",fontsize=16)
#plt.savefig("figs/Stats-Cifar-10.pdf")
#plt.show()
    



