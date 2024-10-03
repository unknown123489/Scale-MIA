# This is the attack main file when the DP mechanism is applied

import argparse
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
import math

from nets.CNNencoder import Encoder
from nets.CNNdecoder import Decoder
from nets.CNNclassifier import Classifier
from opacus import PrivacyEngine

activation = {}
def get_activation(name):
    def hook(model, input, output):
#        activation[name] = output.detach()
        activation[name] = output
    return hook

def batch_retrival(fcweightgrad, fcbiasgrad):
    original_data=torch.zeros(1024,64*4*4)
    for i in range(1024-1):
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
    recover_number=torch.nansum(PSNR>18)
    avg_score=torch.nansum(score[torch.lt(score, 0.03)])/torch.nansum(score<0.03)
    avg_PSNR=torch.nansum(PSNR[torch.gt(PSNR,18)])/torch.nansum(PSNR>18)
    return recover_number, avg_score, avg_PSNR                  

# This function realizes the very important equation 6 in the paper
def input_retrival(fcweightgrad, fcbiasgrad, locations, batch_size):
    original_data=torch.zeros(batch_size, 64 * 4 * 4)
    for i in range(batch_size):
        if i==0:
            index=int(locations[batch_size-i-1])
            original_data[i,:]=fcweightgrad[index, :]/fcbiasgrad[index,:]
        else:
            index=int(locations[batch_size-i-1])
            if index==1024-1:
                original_data[i,:]=fcweightgrad[index, :]/fcbiasgrad[index,:]
            else:
                original_data[i,:]=(fcweightgrad[index, :]-fcweightgrad[index+1, :])/(fcbiasgrad[index,:]-fcbiasgrad[index+1,:]+1e-8)
            
    return original_data

def attack(weight, bias, original_data, l1output, batch_size, idx):
    locations=np.zeros(batch_size)
    for i in range(batch_size):
        locations[i]=1023-(l1output[i,:]<0).count_nonzero()
    sorted_list=np.argsort(locations)
    sorted_location=np.sort(locations)
    recovered_latent=input_retrival(weight, bias, sorted_location, batch_size)
    recovered_data=decoder(recovered_latent.to(device).view(batch_size, 64, 4, 4))   
    round_num, round_mse_score, round_PSNR_score = PSNR_cal(recovered_data, original_data, locations, batch_size)   
    if idx==test_rounds-2:
        torch.save(original_data, "data/data_batch.pt")
        torch.save(recovered_data, "data/recoved.pt")
        torch.save(sorted_list, "data/list.pt")

    return round_num, round_mse_score, round_PSNR_score

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--test_rounds", type=int, default=5)
    parser.add_argument("--client_num", type=int, default=1)
    parser.add_argument("--local_epoch", type=int, default=1)
    parser.add_argument("--all_epoch", type=int, default=1)
    parser.add_argument("--delta", type=float, default=1e-4)
    parser.add_argument("--epsilon", type=float, default=1)
    args = parser.parse_args()

    return args

if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    args=get_args()
    batch_size = args.batch_size
    test_rounds = min(args.test_rounds, int(10000/batch_size))
#    test_rounds = 1
    client_num = args.client_num
    client_size = int(batch_size/client_num)
    all_epoch = args.all_epoch
    local_epoch = args.local_epoch
    epsilon=args.epsilon
    delta=args.delta
    
    
    prev_acc = 0
    transform = transforms.Compose([transforms.ToTensor(),])
    train_dataset = torchvision.datasets.CIFAR10(root='./train', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./test', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    encoder=Encoder()
    encoder.load_state_dict(torch.load("models/encoder_model_100_new.pkl"))
    encoder.to(device)
    decoder=Decoder()
    decoder.load_state_dict(torch.load("models/decoder_model_100_new.pkl"))
    decoder.to(device)
    
    malicious_weight=1/(64*4*4)*torch.ones(1024, 64*4*4).to(device).float()
    malicious_layer2_weight=1/1024*torch.ones(512,1024).to(device).float()
    malicious_layer2_bias=1/512*torch.ones(512).to(device).float()
    malicious_layer3_weight=torch.rand(10,512).to(device).float()
    malicious_layer3_bias=torch.rand(10).to(device).float()
    bins=torch.load("data/latent_small_bins_100.pt")
    malicious_bias=-torch.from_numpy(bins).to(device).float()
    
    number=0
    mse_score=torch.zeros(test_rounds*all_epoch)
    PSNR_score=torch.zeros(test_rounds*all_epoch)
    test_weight=torch.zeros(1024, 64*16).to(device)
    torch.set_printoptions(precision=8)
    gradient_list=torch.zeros(client_num, 1024, 64*16)
    
    for current_epoch in range(all_epoch):
        for idx, (train_x, train_label) in enumerate(test_loader):
            aggregated_weight=torch.zeros(1024, 64*16)
            aggregated_bias=torch.zeros(1024, 64*16)
            gradient_weight=torch.zeros(1024, 64*16).to(device)
            gradient_bias=torch.zeros(1024, 64*16).to(device)
            for j in range(client_num):
                model = Classifier().to(device)
                model.conv1.weight.data=encoder.conv1.weight.data.to(device).clone().detach()
                model.conv2.weight.data=encoder.conv2.weight.data.to(device).clone().detach()
                model.conv3.weight.data=encoder.conv3.weight.data.to(device).clone().detach()
                model.conv1.bias.data=encoder.conv1.bias.data.to(device).clone().detach()
                model.conv2.bias.data=encoder.conv2.bias.data.to(device).clone().detach()
                model.conv3.bias.data=encoder.conv3.bias.data.to(device).clone().detach()
                model.fc1.weight.data = malicious_weight.clone().detach()
                model.fc1.bias.data = malicious_bias.clone().detach()
                model.fc2.weight.data = malicious_layer2_weight.clone().detach()
                model.fc2.bias.data = malicious_layer2_bias.clone().detach()
                model.fc3.weight.data = malicious_layer3_weight.clone().detach()
                model.fc3.bias.data = malicious_layer3_bias.clone().detach()
                model.train()
                sgd = SGD(model.parameters(), lr=0.1)
                privacy_engine = PrivacyEngine()
                model, sgd, test_loader = privacy_engine.make_private_with_epsilon(
                    module=model,
                    optimizer=sgd,
                    data_loader=test_loader,
                    target_epsilon=epsilon,
                    epochs=20,
                    target_delta=delta,
#                    noise_multiplier=1.1,
                    max_grad_norm=10,)
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
                    gradient_bias=model.fc1.bias.grad.data.view(1024,1).expand(1024, 64*16)
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
                aggregated_bias=aggregated_bias+model.fc1.bias.data.cpu().view(1024,1).expand(1024, 64*16)/client_num     
                torch.cuda.empty_cache()
            estimated_weight_gradient=-(aggregated_weight-1/(64*4*4)*torch.ones(1024, 64*4*4).float())/(local_epoch*0.1)
            estimated_bias_gradient=-(aggregated_bias+torch.from_numpy(bins).float().view(1024,1).expand(1024, 64*16))/(local_epoch*0.1)
            original_data=train_x.to(device)
            round_num, round_mse_score, round_PSNR_score=attack(gradient_weight, gradient_bias, original_data, l1output, batch_size, idx)
#            round_num, round_mse_score, round_PSNR_score=attack(estimated_weight_gradient.to(device), estimated_bias_gradient.to(device), original_data, l1output, batch_size, idx)
            number=number+round_num.item()
            mse_score[current_epoch*test_rounds+idx]=round_mse_score.item()
            PSNR_score[current_epoch*test_rounds+idx]=round_PSNR_score.item()
                
            if idx==test_rounds-1:
                break

    print("number", number/(test_rounds*batch_size*all_epoch))
    print("mse score", torch.nansum(mse_score)/torch.nansum(mse_score<0.03))
    print("PSNR score", torch.nansum(PSNR_score)/torch.nansum(PSNR_score>1))     
    
    
    
    
        
