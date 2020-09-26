# Vehicle Re-Identification using Track-to-track ranking of deep latent representation of vehicles

This repository is related to the publication "Improving Vehicle Re-Identification using CNN Latent Spaces: Metrics Comparison and Track-to-track Extension" (https://arxiv.org/abs/1910.09458). This paper is a postprint of the paper submitted and accepted to IET Computer Vision (https://digital-library.theiet.org/content/journals/iet-cvi).

## Vehicle Re-Identification using CNN latent representation

### Latent representation extraction

We defined track of vehicle $T_k$ as a set of $N_k$ images of a vehicle recorded by a given camera. 

$$T_k=\{I_{k,1}, ..., I_{k,N_k}\}$$


For a given image $I_{k,i}\in \mathbb{R}^{n\times m}$, we extract its latent representation (LR) $L_{k,i} \in \mathbb{R}^{f}$ by projecting it in the latent space of a neural network $\mathcal{N}$ (in our experiments, the second-to-last layer of a CNN).

We construct the matrix $\mathbf{L}_{k}=[L_{k,1}, ..., L_{k,N_k}] \in \mathbb{R}^{f\times N_k}$, the LR of the track $T_k$ as a concatenation of the LR of the $N_k$ images of the track.

![alt](img/lr_extraction_one.png)

### T2T Ranking procedure

Given a distance metric $d(.)$, a query track $T_q$, and a set of test tracks $\mathcal{T}=\{T_1, ..., T_{n_t}\}$

The track-to-track ranking (T2T) process consists in ranking every track of $\mathcal{T}$ to construct an ordered set $\tilde{\mathcal{T}}_q = \{T_{q,1}, ..., T_{q,N_t}\}$, such that a track $T_{q,i}$ is the $i^{th}$ nearest track from the query according to the distance function $d(.)$, $T_{q, 1}$ being the first match (i.e. the nearest) and $T_{q, N_t}$, being the last (i.e. the farthest).

![alt](img/LR_extraction_2.png)



## The package ```vehicle_reid```

### Dependencies
- numpy==1.19.2
- torchvision==0.7.0
- torch==1.6.0
- scikit_learn==0.23.2

The python package ```vehicle_reid``` contains code for :
1. Extract the latent representation of images of vehicle using the second-to-last layer of our CNN fine-tuned in the task of vehicle recognition as proposed in our paper. The CNN considered here is based on the DenseNet201 (https://arxiv.org/abs/1608.06993) architectures which has been fine-tuned using the VeRI dataset (https://github.com/VehicleReId/VeRidataset). Corresponding weights are given in ```"data/cnn_weights/VeRI_densenet_ft50.pth"```
2. Compute the Ranking Vehicle Re-identification between tracks of vehicle using the various distance metrics studied in the paper. 
3. Compute the performance metrics rank1 rank5 and mAP.


The module ```vehicle_reid``` is composed of 3 modules 
- ```latent_representation.py```
    - Extract latent representation (LR) of each track of vehicle -> return a json file containing the LR for each track
- ```ranking.py```
    - Compute the ranking for each query track -> return a json file containing the ranking for each query track
- ```performance.py```
    - Compute the performance metrics. namely rank1 rank5 and mAP (See paper for details)



### Running example 
The directory ```data``` contains data to test the module ```vehicle_reid```. Note that to perform the VeRI experiments presented on the paper, you'll need the VeRI dataset which can be found, by simple request to authors, here : https://github.com/JDAI-CV/VeRidataset

- data/cnn_weights/VeRI_densenet_ft50.pth : Pre-trained weights for the DenseNet201 architecture. The model has been trained to classify vehicles of training set the VeRI dataset. Only its latent space (the second-to-last layer) is used to extract features
- data/image_sample : some VeRI tracks of vehicle (splitted in query and test). 

``` 
python3 run_example.py
```








