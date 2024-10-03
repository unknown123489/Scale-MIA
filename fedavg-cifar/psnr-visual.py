import matplotlib.pyplot as plt 
import numpy as np     
import torch  

psnr=torch.load("data/psnr-512.pt").detach().numpy()
print(psnr)

plt.figure(figsize=(7.2,7))
plt.hist(psnr, bins=32, color="b")
plt.xticks(fontsize=24)
plt.yticks([0,8,16,24,32,40,48],[0,8,16,24,32,40,48],fontsize=24)
plt.xlabel("PSNR Score",fontsize=24)
plt.ylabel("Number",fontsize=24)
plt.savefig("figs/psnr_dist_512.pdf")