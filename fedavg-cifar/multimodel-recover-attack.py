# This is the attack main file

import numpy as np
import torch
from torchvision.datasets import mnist
from torch.nn import CrossEntropyLoss
from torch.nn import MSELoss
from torch.optim import SGD
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
from torch.nn import Module
from torch import nn
import matplotlib.pyplot as plt
import argparse
import time
import copy

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--test_rounds", type=int, default=5)
parser.add_argument("--client_num", type=int, default=1)
parser.add_argument("--local_epoch", type=int, default=1)
parser.add_argument("--all_epoch", type=int, default=1)
parser.add_argument("--model_name", default="CNN")
args = parser.parse_args()

name=args.model_name

if name=="CNN":
    from nets.CNNencoder import Encoder
    from nets.CNNdecoder import Decoder
    from nets.CNNclassifier import Classifier
    bin=1024
    features=64
    map_size=4
    l2=512
elif name=="Alexnet":
    from nets.Alexnetencoder import Encoder
    from nets.Alexnetdecoder import Decoder
    from nets.Alexnet import Alexnet
    bin=512
    features=256
    map_size=4
    l2=120
elif name=="Resnet":
    from nets.Resnetencoder import *
    from nets.Resnetdecoder import *
    from nets.Resnet import *
    bin=512
    features=256
    map_size=4
    l2=120
elif name=="Vit":
    from nets.Vitencoder import *
    from nets.Vitdecoder import *
    from nets.Vit import *
    bin=512
    features=512
    map_size=1
    l2=120
elif name=="Vggnet":
    from nets.Vggnetencoder import Encoder
    from nets.Vggnetdecoder import Decoder
    from nets.Vggnet import VGG
    bin=1024
    features=512
    map_size=4
    l2=200


activation = {}
def get_activation(name):
    def hook(model, input, output):
#        activation[name] = output.detach()
        activation[name] = output
    return hook

def batch_retrival(fcweightgrad, fcbiasgrad):
    original_data=torch.zeros(bin,features*map_size*map_size)
    for i in range(bin-1):
        original_data[i,:]=(fcweightgrad[i, :]-fcweightgrad[i+1, :])/(fcbiasgrad[i,:]-fcbiasgrad[i+1,:])
    return original_data

def PSNR_cal(recovered_data, original_data, locations, batch_size):
    mse_fn=MSELoss()   
    sorted_index=np.argsort(locations)
    
    score=torch.zeros(batch_size)
    PSNR=torch.zeros(batch_size)
    
    for i in range(batch_size):
        score[i]=mse_fn(original_data[sorted_index[batch_size-1-i],:],recovered_data[i,:])
        PSNR[i]=-10*torch.log10(score[i])
#    print(PSNR)
    recover_number=torch.nansum(PSNR>18)
    avg_score=torch.nansum(score[torch.lt(score, 0.03)])/torch.nansum(score<0.03)
    avg_PSNR=torch.nansum(PSNR[torch.gt(PSNR,18)])/torch.nansum(PSNR>18)
    return recover_number, avg_score, avg_PSNR                  

# This function realize the very important equation 6 in the paper
def input_retrival(fcweightgrad, fcbiasgrad, locations, batch_size):
    original_data=torch.zeros(batch_size, features*map_size*map_size)
    for i in range(batch_size):
        if i==0:
            index=int(locations[batch_size-i-1])
            original_data[i,:]=fcweightgrad[index, :]/fcbiasgrad[index,:]
        else:
            index=int(locations[batch_size-i-1])
            if index==bin-1:
                original_data[i,:]=fcweightgrad[index, :]/fcbiasgrad[index,:]
            else:
                original_data[i,:]=(fcweightgrad[index, :]-fcweightgrad[index+1, :])/(fcbiasgrad[index,:]-fcbiasgrad[index+1,:]+1e-8)
            
    return original_data

def attack(weight, bias, original_data, l1output, batch_size, idx):
    locations=np.zeros(batch_size)
    for i in range(batch_size):
        locations[i]=bin-1-(l1output[i,:]<0).count_nonzero()
    sorted_list=np.argsort(locations)
    sorted_location=np.sort(locations)
    recovered_latent=input_retrival(weight, bias, sorted_location, batch_size)
    recovered_data=decoder(recovered_latent.to(device).view(batch_size, features, map_size, map_size))   
    round_num, round_mse_score, round_PSNR_score = PSNR_cal(recovered_data, original_data, locations, batch_size)   
    return round_num, round_mse_score, round_PSNR_score

if __name__ == '__main__':
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    
    batch_size = args.batch_size
    test_rounds = min(args.test_rounds, int(10000/batch_size))
#    test_rounds = 1
    client_num = args.client_num
    client_size = int(batch_size/client_num)
    all_epoch = args.all_epoch
    local_epoch = args.local_epoch
    prev_acc = 0
    
    transform = transforms.Compose([transforms.ToTensor(),])
    train_dataset = torchvision.datasets.CIFAR10(root='./train', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./test', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
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
            ) 
    elif name=="Resnet":
        encoder=Encoder(ResidualBlock, [2, 2, 2])
    else:
        encoder = Encoder()
    encoder.load_state_dict(torch.load("models/encoder_model_"+name+".pkl"))
    encoder.to(device)
    decoder=Decoder()
    decoder.load_state_dict(torch.load("models/decoder_model_"+name+".pkl"))
    decoder.to(device)
    
    malicious_weight=1/(features*map_size*map_size)*torch.ones(bin, features*map_size*map_size).to(device).float()
    malicious_layer2_weight=1/bin*torch.ones(l2,bin).to(device).float()
    malicious_layer2_bias=1/l2*torch.ones(l2).to(device).float()
    malicious_layer3_weight=torch.rand(10,l2).to(device).float()
    malicious_layer3_bias=torch.rand(10).to(device).float()
    bins=torch.load("data/latent_small_bins_"+name+".pt")
    malicious_bias=-torch.from_numpy(bins).to(device).float()
    
    number=0
    mse_score=torch.zeros(test_rounds*all_epoch)
    PSNR_score=torch.zeros(test_rounds*all_epoch)
    test_weight=torch.zeros(bin, features*map_size*map_size).to(device)
    torch.set_printoptions(precision=8)
    start_time=time.time()
    for current_epoch in range(all_epoch):
        for idx, (train_x, train_label) in enumerate(test_loader):
            aggregated_weight=torch.zeros(bin, features*map_size*map_size)
            aggregated_bias=torch.zeros(bin, features*map_size*map_size)
            gradient_weight=torch.zeros(bin, features*map_size*map_size).to(device)
            gradient_bias=torch.zeros(bin, features*map_size*map_size).to(device)
            for j in range(client_num):
                if name=="CNN":
                    model = Classifier().to(device)
                    model.conv1.weight.data=encoder.conv1.weight.data.to(device).clone().detach()
                    model.conv2.weight.data=encoder.conv2.weight.data.to(device).clone().detach()
                    model.conv3.weight.data=encoder.conv3.weight.data.to(device).clone().detach()
                    model.conv1.bias.data=encoder.conv1.bias.data.to(device).clone().detach()
                    model.conv2.bias.data=encoder.conv2.bias.data.to(device).clone().detach()
                    model.conv3.bias.data=encoder.conv3.bias.data.to(device).clone().detach()
                elif name=="Alexnet":
                    model = Alexnet().to(device)
                    model.features=copy.deepcopy(encoder.features)
                elif name=="Vggnet":
                    model = VGG().to(device)
                    model.features=copy.deepcopy(encoder.features)
                elif name=="Resnet":
                    model = ResNet(ResidualBlock, [2, 2, 2]).to(device)
                    model.conv=copy.deepcopy(encoder.conv)
                    model.layer1=copy.deepcopy(encoder.layer1)
                    model.layer2=copy.deepcopy(encoder.layer2)
                    model.layer3=copy.deepcopy(encoder.layer3)
                    model.layer4=copy.deepcopy(encoder.layer4)
                elif name=="Vit":
                    model = ViT(image_size=32,patch_size=8,num_classes=10,channels=3,dim=512,depth=2,heads=1,mlp_dim=512,dropout=0,emb_dropout=0).to(device)
                    model.to_patch_embedding = copy.deepcopy(encoder.to_patch_embedding)
                    model.pos_embedding = copy.deepcopy(encoder.pos_embedding)
                    model.cls_token = copy.deepcopy(encoder.cls_token)
                    model.pos_embedding = copy.deepcopy(encoder.pos_embedding)
                    model.transformer = copy.deepcopy(encoder.transformer)
                    model.to_latent = copy.deepcopy(encoder.to_latent)
                model.fc1.weight.data = malicious_weight.clone().detach()
                model.fc1.bias.data = malicious_bias.clone().detach()
                model.fc2.weight.data = malicious_layer2_weight.clone().detach()
                model.fc2.bias.data = malicious_layer2_bias.clone().detach()
                model.fc3.weight.data = malicious_layer3_weight.clone().detach()
                model.fc3.bias.data = malicious_layer3_bias.clone().detach()
                model.train()
                sgd = SGD(model.parameters(), lr=0.1)
                loss_fn = CrossEntropyLoss()
                if j==0:
                    handle=model.fc1.register_forward_hook(get_activation('fc1'))
                    predict_y = model(train_x.float().to(device))
                    l1output=activation['fc1'].view(batch_size,-1)
                    handle.remove()
                    sgd.zero_grad()
                    loss = loss_fn(predict_y, train_label.long().to(device))
                    loss.backward()
                    gradient_weight=model.fc1.weight.grad.data
                    gradient_bias=model.fc1.bias.grad.data.view(bin,1).expand(bin, features*map_size*map_size)
                for k in range(local_epoch):
                    train_x_local = train_x[j*client_size: (j+1)*client_size,:,:,:].to(device)
                    train_label_local = train_label[j*client_size: (j+1)*client_size].to(device)
                    sgd.zero_grad()
                    predict_y_local = model(train_x_local.float())            
                    loss = loss_fn(predict_y_local, train_label_local.long())
                    loss.backward()
                    sgd.step()
                mse_fn=MSELoss()
                aggregated_weight=aggregated_weight+model.fc1.weight.data.cpu()/client_num
                aggregated_bias=aggregated_bias+model.fc1.bias.data.cpu().view(bin,1).expand(bin, features*map_size*map_size)/client_num     
                torch.cuda.empty_cache()
            estimated_weight_gradient=-(aggregated_weight-1/(features*map_size*map_size)*torch.ones(bin, features*map_size*map_size).float())/(local_epoch*0.1)
            estimated_bias_gradient=-(aggregated_bias+torch.from_numpy(bins).float().view(bin,1).expand(bin, features*map_size*map_size))/(local_epoch*0.1)
            original_data=train_x.to(device)
            round_num, round_mse_score, round_PSNR_score=attack(estimated_weight_gradient.to(device), estimated_bias_gradient.to(device), original_data, l1output, batch_size, idx)
            number=number+round_num.item()
            mse_score[current_epoch*test_rounds+idx]=round_mse_score.item()
            PSNR_score[current_epoch*test_rounds+idx]=round_PSNR_score.item()
                
            if idx==test_rounds-1:
                end_time=time.time()
                break
            
    print("number", number/(test_rounds*batch_size*all_epoch))
    print("mse score", torch.nansum(mse_score)/torch.nansum(mse_score<0.03))
    print("PSNR score", torch.nansum(PSNR_score)/torch.nansum(PSNR_score>1))
    print("Attack time", (end_time-start_time)/(test_rounds*all_epoch))   
    
    
    
    
        
