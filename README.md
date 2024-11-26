# Scale-MIA Artifact
This repository contains the software artifact of the NDSS'25 paper "Scale-MIA: A Scalable Model Inversion Attack against Secure Federated Learning via Latent Space Reconstruction." Please check the following Bibtex for reference.
```
@misc{shi2023scalemiascalablemodelinversion,
      title={Scale-MIA: A Scalable Model Inversion Attack against Secure Federated Learning via Latent Space Reconstruction}, 
      author={Shanghao Shi and Ning Wang and Yang Xiao and Chaoyu Zhang and Yi Shi and Y. Thomas Hou and Wenjing Lou},
      year={2023},
      eprint={2311.05808},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2311.05808}, 
}
```

## Abstract
Federated learning (FL) has been widely regarded as a privacy-preserving distributed learning paradigm. However, in this work, we propose a novel model inversion attack named Scale-MIA that breaks this property to have the malicious server reconstruct private local samples hold by the clients, even when the FL system is protected by the state-of-the-art secure aggregation mechanism. To achieve this, the attacker modifies the global model's parameters before sending it to the clients. The clients, without finding the stealthy and unnoticeable modifications, will train the local models according to the adversarial global model and send their model updates back. Then the attacker, i.e. the server, receives the aggregation of these model updates when the secure aggregation mechanism is in place, and can reconstruct local training samples from them with an efficient analytical method. Compared to the existing work, Scale-MIA is stealthier without the need to modify the model architecture; more efficient, which does not rely on any costly optimization process during the attack phase; and can reconstruct large-scale local sample batches. In this artifact, we show how we implement our attack based on the PyTorch framework. We will demonstrate how the attack is implemented in detail and how the results and figures in the paper are generated. 
Particularly, our artifact offers flexible interfaces, allowing users to modify experiment settings and validate the results of any experiments.

## Get Started

**Requirements:**
This implementation is Pytorch-based and requires the following packages:
- Python 3.8.10
- torch 2.0.1
- torchvision 0.15.2
- numpy 1.24.4
- matplotlib 3.7.4
- opacus 1.3.0
- einops 0.8.0
- datasets 3.0.1
- scipy 1.14.1

We recommend users conduct all experiments on a Ubuntu-based machine. Users can install any missing packages via the following command:
```
pip3 install *packagename*
```
If the pip3 package itself is missing, users can use the following command to install it:
```
sudo apt install python3-pip
```

**Download:**
After obtaining all necessary packages, users can download our code with the following command:
```
git clone https://github.com/unknown123489/Scale-MIA.git
```

**Repository Structure:**
This repository is organized according to the experiment datasets and target systems. Each folder contains all necessary code, data, and results for a specific dataset and system. For example, the fedavg-tinyimagenet/ folder contains all code, data, and results related to the TinyImageNet dataset targeting the FedAVG system. Users can enter each folder separately to evaluate the experiment results independently. 
- fedavg-tinyimagenet/: This folder contains all code and image reconstruction results related to the TinyImageNet dataset in Table 4, Figure 4, and Figure 10. This folder also provides the PSNR distributions for different reconstruction batch sizes, corresponding to Figure 5.
- fedavg-fmnist/: This folder contains all code and image reconstruction results related to the FashionMNIST dataset in Table 4, Figure 4, and Figure 8.
- fedavg-hmnist/: This folder contains all code and image reconstruction results related to the HMNIST dataset in Table 4, Figure 4, and Figure 9.
- fedavg-cifar/: This folder contains code and image reconstruction results for the CIFAR-10 dataset in Table 4, Figure 4, and Figure 7. This folder also contains Scale-MIA's performance under data deficiency and bias settings, as shown in Tables 6 and 7. The code and experiment results for Scale-MIA under different models (including Resnet, Vggnet, Alexnet, and Vit) are also contained.
- dp-cifar/: This folder contains the code and experiment results when the DP-SGD mechanism is applied as an additional defense, producing the results in Table 8.

## Artifact Evaluation
**Image Reconstruction Results:**
For all folders, there are mainly three Python files realizing the Scale-MIA attack, including the fedavg-adv-train.py, fedavg-para-gen.py, and fedavg-recover-attack.py. 
* The first one is in charge of training a surrogate autoencoder (Steps 1 to 3 in the attack flow). (**Optional**)
* The second one is in charge of generating attack parameters (Steps 4 to 5 in the attack flow). (**Optional**)
* The third one is the main attack file, and users can change the parameters within this file such as batch_size, client_num, rounds, etc (Steps 6 to 8 in the attack flow).
* Users can go into each folder (such as the fedavg-tinyimagenet) and execute them one by one to go through the whole attack process.
```
cd fedavg-tinyimagenet
```
```
python fedavg-adv-train.py
```
The loss function of the training process (i.e. loss_trend.png) can be found in the figs/ folder. 

```
python fedavg-para-gen.py
```
```
python fedavg-recover-attack.py
```
However, we recommend the user focus **solely on the third one** to evaluate the attack performance. This is because the first and second files involve the training of the surrogate autoencoder and the generation of attack parameters and thus consume a lot of time and computation resources. This training process usually consumes tens of execution minutes and large GPU memory. Furthermore, our attack only requires the training process to be conducted once, while the attack phase itself can be conducted afterward without limitation, i.e. "train once and attack multiple times". Therefore, to avoid the training cost, we provide all necessary **fine-tuned** and **pre-trained** models produced by the first two files (in the models/ and data/ folders), and the users can only execute the third file to check the results repeatedly. Users can specify different arguments for the attack such as the reconstruction batch size (--batch_size) and test round (--test_rounds).
```
cd fedavg-tinyimagenet
```
```
python fedavg-recover-attack.py --batch_size=64 --test_round=5
```
We provide the following arguments for users to evaluate different attack settings (according to Table 4). All of them have default values and users do not need to specify them all during each trial.
- Batch size (--batch_size): Refers to the reconstruction batch size (Default value: 128). (* Prefered to be 64, 128, 256, 512.)
- Test rounds (--test_rounds): Refers to the number of reconstructed batches (Default value: 10). 
- Client number (--client_num): Refers to the number of FL clients (Default value: 8). (* Prefered to be 4, 8, 16.)
- Local epoch (--local_epoch): Refers to the number of local epochs the client conducts (Default value: 1).

For each folder, our code will automatically download the corresponding dataset in the **/train** and **/test** folders when executed for the first time. The execution results will demonstrate the following information:
- Reconstruction rate: The ratio of successfully reconstructed images.
- PSNR score: Peak signal-to-noise ratio score. The score indicates the reconstruction performance and higher scores indicate better performance.
- MSE score: Mean square error score. Lower scores indicate better performance.
- Attack time: Execution time of the attack main file. Scale-MIA is an efficient attack and can be finished within one second 

**Visualize Reconstructed Images:** Users can visualize the reconstructed images and original ones by executing the following commands within a specific folder (e.g. fedavg-tinyimagenet).
```
cd fedavg-tinyimagenet
```
```
python visual.py
```
The original and reconstructed images are shown in the figs/ folder. The two figures show a batch of 64 original images and the reconstructed ones. Most of the images are reconstructed with high quality.

Furthermore, users can obtain the PSNR distribution figure (Figure 5 in the paper) with the following command.
```
python psnr-visual.py
```

**Data Deficiency Results:** We provide experiment results under different auxiliary dataset settings assuming the attacker cannot collect enough auxiliary samples. We consider the auxiliary dataset to own 1%, 3%, 10%, and 100% of the training set and aim to reconstruct samples in the non-intersected test set, which is consistent with the settings in Table 6. We specify the --aux argument for users to validate the results in the paper within the fedavg-cifar/ folder. Note that the previous arguments still work.
```
cd fedavg-cifar
```
```
python fedavg-recover-attack.py --aux=10
```
- Auxiliary dataset (--aux): Refers to the portion of samples owned by the auxiliary dataset. (Default value: 100)(Can be 1, 3, 10, 100.)


**Data Bias Results:** We provide experiment results under biased data settings within the fedavg-cifar/ folder (i.e. the attacker only owns a specific number of classes and we set it as 3 in this implementation). The results can be validated via the following commands:
```
cd fedavg-cifar
```
```
python targeted-recover-attack.py
```
The whole biased attack process can be evaluated by executing the three files in order:
```
python targeted-adv-train.py
```
```
python targeted-para-gen.py
```
```
python targeted-recover-attack.py
```

**Regarding Different Model Architectures:** We provide implementations and experiment results for different ML model architectures within the fedavg-cifar/ folder, corresponding to Table 5 in the paper. The user can specify the following additional argument to check the attack performance. 
```
cd fedavg-cifar
```
```
python multimodel-recover-attack.py --model_name=Resnet
```
- Model name (--model_name): Refers to the name of the ML model. (Default value: CNN)(Candidate values: CNN, Vggnet, Alexnet, Resnet, and Vit.)

**Differential Privacy Results:**
Users need to install Opacus ([https://github.com/pytorch/opacus]), an open-source implementation of the differential privacy mechanisms on the Pytorch platform before going to the dp-cifar/ folder.
```
pip3 install opacus
```
Users can run the following commands to assess Scale-MIA's performance while the system is protected by the DP mechanisms. Users can specify different epsilon and delta arguments for the DP mechanism.
```
cd dp-cifar
```
```
python dp-recover-attack.py --delta=1e-4 --epsilon=1
```
- Epsilon (--epsilon): Refers to the epsilon parameter of the DP mechanism. (Default value: 1e-4)
- Delta (--delta): Refers to the delta parameter of the DP mechanism. (Default value: 1)

<!---
## Benchmarks:
For the benchmarks we compared in this work, users are referred to the following open-source implementations:
- https://github.com/mit-han-lab/dlg
- https://github.com/JonasGeiping/invertinggradients
- https://github.com/JonasGeiping/breaching
-->

