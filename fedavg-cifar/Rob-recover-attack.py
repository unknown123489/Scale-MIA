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
from nets.ROBclassifier import Classifier
import time

activation = {}
def get_activation(name):
    def hook(model, input, output):
#        activation[name] = output.detach()
        activation[name] = output
    return hook

def model_initilization(model):
    malicious
    model.conv3.weight.data = malicious_weight.clone().detach()
    
def PSNR_cal(recovered_data, original_data, locations, batch_size):
    mse_fn=MSELoss()   
    sorted_index=np.argsort(locations)
    
    score=torch.zeros(batch_size)
    PSNR=torch.zeros(batch_size)
    
    for i in range(batch_size):
        score[i]=mse_fn(original_data[sorted_index[batch_size-1-i],:],recovered_data[i,:])
        PSNR[i]=-10*torch.log10(score[i])
#    print(score)
#    print(PSNR)
    recover_number=torch.nansum(PSNR>18)
    avg_score=torch.nansum(score[torch.lt(score, 0.03)])/torch.nansum(score<0.03)
    avg_PSNR=torch.nansum(PSNR[torch.gt(PSNR,18)])/torch.nansum(PSNR>18)
    return recover_number, avg_score, avg_PSNR                  


# This function need to be modified
def input_retrival(fcweightgrad, fcbiasgrad, locations, batch_size):
    original_data=torch.zeros(batch_size, 3 * 32 * 32)
    for i in range(batch_size):
        if i==0:
            index=int(locations[batch_size-i-1])
            original_data[i,:]=fcweightgrad[index, :]/fcbiasgrad[index,:]
        else:
            index=int(locations[batch_size-i-1])
            if index==8*8*32-1:
                original_data[i,:]=fcweightgrad[index, :]/fcbiasgrad[index,:]
            else:
                original_data[i,:]=(fcweightgrad[index, :]-fcweightgrad[index+1, :])/(fcbiasgrad[index,:]-fcbiasgrad[index+1,:]+1e-8)
            
    return original_data

# This function need to be modified
def attack(weight, bias, original_data, l1output, batch_size, idx):
    locations=np.zeros(batch_size)
    for i in range(batch_size):
        locations[i]=8*8*32-1-(l1output[i,:]<0).count_nonzero()
    sorted_list=np.argsort(locations)
    sorted_location=np.sort(locations)
    recovered_data=input_retrival(weight, bias, sorted_location, batch_size)  
    round_num, round_mse_score, round_PSNR_score = PSNR_cal(recovered_data.to(device), original_data, locations, batch_size)   
    if idx==test_rounds-1:
        torch.save(original_data, "data/data_batch.pt")
        torch.save(recovered_data, "data/recoved.pt")
        torch.save(sorted_list, "data/list.pt")
    return round_num, round_mse_score, round_PSNR_score

# This function need to be modified
if __name__ == '__main__':
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    torch.cuda.empty_cache()
    batch_size = 256
#    test_rounds = int(10000/batch_size)
    test_rounds = 8
    client_num = 1
    client_size = int(batch_size/client_num)
    all_epoch = 1
    local_epoch = 1
    prev_acc = 0
    
    transform = transforms.Compose([transforms.ToTensor(),])
    train_dataset = torchvision.datasets.CIFAR10(root='./train', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./test', train=False, download=True, transform=transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
      
    malicious_weight=1/(3*32*32)*torch.ones(32*8*8, 3*32*32).to(device).float()
    malicious_layer2_weight=1/2048*torch.ones(32*8*8,32*8*8).to(device).float()
    malicious_layer2_bias=1/2048*torch.ones(32*8*8).to(device).float()
    bins=torch.load("data/latent_small_bins_rob_100.pt")
    malicious_bias=-torch.from_numpy(bins).to(device).float()
    
    malicious_conv3_weight=torch.rand(64,32,4,4).to(device).float()
    malicious_conv3_bias=torch.rand(64).to(device).float()
    malicious_fc1_weight=torch.rand(1024,1024).to(device).float()
    malicious_fc1_bias=torch.rand(1024).to(device).float()
    malicious_fc2_weight=torch.rand(512,1024).to(device).float()
    malicious_fc2_bias=torch.rand(512).to(device).float()
    malicious_fc3_weight=torch.rand(10,512).to(device).float()
    malicious_fc3_bias=torch.rand(10).to(device).float()
    
    number=0
    mse_score=torch.zeros(test_rounds*all_epoch)
    PSNR_score=torch.zeros(test_rounds*all_epoch)
    torch.set_printoptions(precision=8)
    for current_epoch in range(all_epoch):
        start_time=time.time()
        for idx, (train_x, train_label) in enumerate(test_loader):
            aggregated_weight=torch.zeros(32*8*8, 3*32*32)
            aggregated_bias=torch.zeros(32*8*8, 3*32*32)
            gradient_weight=torch.zeros(32*8*8, 3*32*32).to(device)
            gradient_bias=torch.zeros(32*8*8, 3*32*32).to(device)
            for j in range(client_num):
                model = Classifier().to(device)
                model.ln1.weight.data = malicious_weight.clone().detach()
                model.ln1.bias.data = malicious_bias.clone().detach()
                model.ln2.weight.data = malicious_layer2_weight.clone().detach()
                model.ln2.bias.data = malicious_layer2_bias.clone().detach()
                
                # All layers shall be identical
                model.conv3.weight.data = malicious_conv3_weight.clone().detach()
                model.conv3.bias.data = malicious_conv3_bias.clone().detach()
                model.fc1.weight.data = malicious_fc1_weight.clone().detach()
                model.fc1.bias.data = malicious_fc1_bias.clone().detach()
                model.fc2.weight.data = malicious_fc2_weight.clone().detach()
                model.fc2.bias.data = malicious_fc2_bias.clone().detach()
                model.fc3.weight.data = malicious_fc3_weight.clone().detach()
                model.fc3.bias.data = malicious_fc3_bias.clone().detach()
                
                model.train()
                sgd = SGD(model.parameters(), lr=0.1)
                loss_fn = CrossEntropyLoss()
                if j==0:
                    handle=model.ln1.register_forward_hook(get_activation('ln1'))
                    predict_y = model(train_x.float().to(device))
                    l1output=activation['ln1'].view(batch_size,-1)
                    handle.remove()
                    sgd.zero_grad()
                    loss = loss_fn(predict_y, train_label.long().to(device))
                    loss.backward()
                    gradient_weight=model.ln1.weight.grad.data
                    gradient_bias=model.ln1.bias.grad.data.view(32*8*8,1).expand(32*8*8, 3*32*32)
                for k in range(local_epoch):
                    train_x_local = train_x[j*client_size: (j+1)*client_size,:,:,:].to(device)
                    train_label_local = train_label[j*client_size: (j+1)*client_size].to(device)
                    sgd.zero_grad()
                    predict_y_local = model(train_x_local.float())            
                    loss = loss_fn(predict_y_local, train_label_local.long())
                    loss.backward()
                    sgd.step()
                mse_fn=MSELoss()
                aggregated_weight=aggregated_weight+model.ln1.weight.data.cpu()/client_num
                aggregated_bias=aggregated_bias+model.ln1.bias.data.cpu().view(32*8*8,1).expand(32*8*8, 3*32*32)/client_num     
                torch.cuda.empty_cache()
            estimated_weight_gradient=-(aggregated_weight-1/(3*32*32)*torch.ones(32*8*8, 3*32*32).float())/(local_epoch*0.1)
            estimated_bias_gradient=-(aggregated_bias+torch.from_numpy(bins).float().view(32*8*8,1).expand(32*8*8, 3*32*32))/(local_epoch*0.1)
            original_data=train_x.to(device).view(batch_size,-1)
#            round_num, round_mse_score, round_PSNR_score=attack(gradient_weight.to(device), gradient_bias.to(device), original_data, l1output, batch_size, idx)
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
    
    
    
    
        
