# This file is the training process of the surrogate autoencoder, corresponds to attack steps 1 to 3
# We have plotted the training loss in the figs/loss_trend figure
# For convenience, we have already conducted the training process and stored the already trained parameters in the models folder
# The user can directly execute the attack main file for evaluation
# This file is for the targeted attack. 
# In this example, we assume the attacker collects the ship, dog, and frog classes out of the 10 classes in the CIFAR-10 dataset

import matplotlib.pyplot as plt 
import numpy as np 
import random 
import torch
import torchvision
from torchvision import transforms
from torchvision.datasets import mnist
from torch.utils.data import Dataset,DataLoader
from torchvision.transforms import ToTensor
from torch import nn
from torch.nn import MSELoss
import torch.nn.functional as F
import torch.optim as optim
from nets.CNNencoder import Encoder
from nets.CNNdecoder import Decoder
from torchvision.datasets import CIFAR10

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

print(device)

# The sample_num parameter controls the size of the auxiliary dataset
# The user can change it to different numbers for different auxiliary dataset sizes

sample_num=15000
batch_size = 50
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


encoder = Encoder().to(device)
decoder = Decoder().to(device)

params_to_optimize = [
    {'params': encoder.parameters()},
    {'params': decoder.parameters()}
]

lr=0.001
optimizer = torch.optim.Adam(params_to_optimize, lr=lr, weight_decay=1e-05)

loss_fn = MSELoss()
all_epoch = 250
prev_acc = 0
train_loss=[]

for current_epoch in range(all_epoch): 
    encoder.train()
    decoder.train()
    epoch_loss=0
    for idx, (train_x, train_label) in enumerate(train_loader):
        if idx<int(sample_num/batch_size):
            train_x=train_x.to(device)
            encoded_data=encoder(train_x)
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

torch.save(encoder.state_dict(), "models/encoder_model_targeted_new.pkl") 
torch.save(decoder.state_dict(), "models/decoder_model_targeted_new.pkl") 
    
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
plt.show()        
    
    
    
    
    
    
    
    

