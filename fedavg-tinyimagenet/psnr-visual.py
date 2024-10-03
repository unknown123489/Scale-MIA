# This file is used to generate the figures in Fig. 5 in the paper
# The user can manually change the batch size to plot different subfigures
# The current batch size is 1024 (psnr-1024)
# The user can change it to 128/256/512
# For convenience, we have already plotted all subfigures and they can be found in the figs folder

import matplotlib.pyplot as plt 
import numpy as np     
import torch  

psnr=torch.load("data/psnr-1024.pt").detach().numpy()
print(psnr)

plt.figure(figsize=(7.2,7))
plt.hist(psnr, bins=40, color="b") #32
plt.xticks(fontsize=24)
plt.yticks([0,10,20,30,40,50,60],[0,10,20,30,40,50,60],fontsize=24)
plt.xlabel("PSNR Score",fontsize=24)
plt.ylabel("Number",fontsize=24)
plt.savefig("figs/psnr_dist_1024.pdf")
