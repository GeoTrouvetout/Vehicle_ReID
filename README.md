# Vehicle Re-Identification using Track-to-track ranking of deep latent representation of vehicles

This repository is related to the publication "Improving Vehicle Re-Identification using CNN Latent Spaces: Metrics Comparison and Track-to-track Extension" (https://arxiv.org/abs/1910.09458). This paper is a postprint of the paper submitted and accepted to IET Computer Vision (https://digital-library.theiet.org/content/journals/iet-cvi).

## Vehicle Re-Identification using CNN latent representation

### Latent representation extraction

We defined track of vehicle <img src="https://render.githubusercontent.com/render/math?math=T_k=\{I_{k,1}, ..., I_{k,N_k}\}"> as a set of <img src="https://render.githubusercontent.com/render/math?math=N_k"> images of a vehicle recorded by a given camera. 


For a given image <img src="https://render.githubusercontent.com/render/math?math=I_{k,i}\in \mathbb{R}^{n\times m}">, we extract its latent representation (LR) <img src="https://render.githubusercontent.com/render/math?math=L_{k,i} \in \mathbb{R}^{f}"> by projecting it in the latent space of a neural network <img src="https://render.githubusercontent.com/render/math?math=\mathcal{N}"> (in our experiments, the second-to-last layer of a CNN).

We construct the matrix <img src="https://render.githubusercontent.com/render/math?math=\mathbf{L}_{k}=[L_{k,1}, ..., L_{k,N_k}] \in \mathbb{R}^{f\times N_k}">, the LR of the track $T_k$ as a concatenation of the LR of the <img src="https://render.githubusercontent.com/render/math?math=N_k"> images of the track.

![alt](img/lr_extraction_one.png)

### I2T/T2T Ranking procedure

Given a distance metric <img src="https://render.githubusercontent.com/render/math?math=d(.)">, a query track <img src="https://render.githubusercontent.com/render/math?math=T_q">, and a set of test tracks <img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}=\{T_1, ..., T_{n_t}\}">

The track-to-track ranking (T2T) process consists in ranking every track of <img src="https://render.githubusercontent.com/render/math?math=\mathcal{T}"> to construct an ordered set <img src="https://render.githubusercontent.com/render/math?math=\tilde{\mathcal{T}}_q = \{T_{q,1}, ..., T_{q,N_t}\}">, such that a track <img src="https://render.githubusercontent.com/render/math?math=T_{q,i}"> is the <img src="https://render.githubusercontent.com/render/math?math=i^{th}"> nearest track from the query according to the distance function <img src="https://render.githubusercontent.com/render/math?math=d(.)">, <img src="https://render.githubusercontent.com/render/math?math=T_{q, 1}"> being the first match (i.e. the nearest) and <img src="https://render.githubusercontent.com/render/math?math=T_{q, N_t}">, being the last (i.e. the farthest).

![alt](img/LR_extraction_2.png)

The image-to-track ranking (I2T) corresponds to the T2T ranking procedure but with a query track composed of only one image <img src="https://render.githubusercontent.com/render/math?math=T_q = I_q">, and its corresponding LR <img src="https://render.githubusercontent.com/render/math?math=L_q"> (only the distance metric d used differs). 

### Distance metric (I2T)

In I2T ranking process the distance <img src="https://render.githubusercontent.com/render/math?math=d(.)"> is computed between a query composed of one image <img src="https://render.githubusercontent.com/render/math?math=L_{q}">, and a test track <img src="https://render.githubusercontent.com/render/math?math=\mathbf{L}_r = \{L_{r, 1}, ..., L_{r, n_t}\}">.


    - MED : Minimal Euclidean Distance

 <img src="https://render.githubusercontent.com/render/math?math=d(L_{q}, \mathbf{L}_r) = \underset{i \in \{1, ..., N_r\}}{min} (|| L_q - L_{r,i} ||_2)">

    - MCD : Minimal Cosine Distance

<img src="https://render.githubusercontent.com/render/math?math=d(L_{q}, \mathbf{L}_r) = \underset{i \in \{1, ..., N_r\}}{min}  (1 - \frac{L_q^\top L_{r, i}}{|| L_q ||_2  || L_{r,i} ||_2} )"> 

    - RSCR : Residual of the Sparse Coding Reconstruction
  
<img src="https://render.githubusercontent.com/render/math?math=d(L_q, \mathbf{L}_r ) = {|| L_q -  \mathbf{L}_r \Gamma_{q,r} ||_2}^2">

with 

<img src="https://render.githubusercontent.com/render/math?math=\Gamma_{q,r} = \underset{\tilde\Gamma_{q,r}}{argmin} ( {|| L_q -  \mathbf{L}_r \tilde{\Gamma}_{q,r} ||_2}^2 %2B
\alpha || \tilde{\Gamma}_{q,r} ||_1 )">



### Aggregation function for T2TP

In track-to-track (T2T) ranking process the distance <img src="https://render.githubusercontent.com/render/math?math=d(.)"> is computed between a query track <img src="https://render.githubusercontent.com/render/math?math=\mathbf{L}_{q} = \{L_{q, 1}, ..., L_{q, n_q}\}"> and a test track <img src="https://render.githubusercontent.com/render/math?math=\mathbf{L}_r = \{L_{r, 1}, ..., L_{r, n_t}\}">. 

- If the distance metric is based on MED or MCD, an aggregation function <img src="https://render.githubusercontent.com/render/math?math=agg(.)"> is used to aggregate the set of I2T distances (<img src="https://render.githubusercontent.com/render/math?math=d_{i2t}(.)">) between each <img src="https://render.githubusercontent.com/render/math?math=L_{q, i}"> of the query and the test track <img src="https://render.githubusercontent.com/render/math?math=\mathbf{L}_r"> : 

<img src="https://render.githubusercontent.com/render/math?math=d(\mathbf{L}_q, \mathbf{L}_r) = agg ( \{ d_{i2t}(L_{q,1}, \mathbf{L}_r), ...,  d_{i2t}(L_{q,n_q}, \mathbf{L}_{r}) \} ) ">

    - min : minimum of distances
    - mean : average of distances
    - med : median of distances
    - mean50 : average of distances between the 50% smallest distances
    - med50 : average of distances between 50% smallest distances

- If the distance is based on RSCR, the distance between <img src="https://render.githubusercontent.com/render/math?math=\mathbf{L}_{q}"> and <img src="https://render.githubusercontent.com/render/math?math=\mathbf{L}_r "> is computed as follows : 


<img src="https://render.githubusercontent.com/render/math?math=d(\mathbf{L}_q, \mathbf{L}_r ) = || \mathbf{L}_q -  \mathbf{L}_r \mathbf{\Gamma}_{q,r} ||_F">

with 

<img src="https://render.githubusercontent.com/render/math?math={\Gamma}_{q_i,r} = \underset{\tilde{\Gamma}_{q_i,r}}{\mathrm{argmin}} ( {|| L_{q,i} -  \mathbf{L}_r \tilde{\Gamma}_{q_i,r} ||_2}^2  %2B \alpha || \tilde{\Gamma}_{q_i,r} ||_1)">

Note : <img src="https://render.githubusercontent.com/render/math?math=||.||_F"> denotes the Frobenius norm

## The package ```vehicle_reid```

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

### Dependencies
- numpy==1.19.2
- torchvision==0.7.0
- torch==1.6.0
- scikit_learn==0.23.2

### Running example 
The directory ```data``` contains data to test the module ```vehicle_reid```. Note that to perform the VeRI experiments presented on the paper, you'll need the VeRI dataset which can be found, by simple request to authors, here : https://github.com/JDAI-CV/VeRidataset

- data/cnn_weights/VeRI_densenet_ft50.pth : Pre-trained weights for the DenseNet201 architecture. The model has been trained to classify vehicles of training set the VeRI dataset. Only its latent space (the second-to-last layer) is used to extract features
- data/image_sample : some VeRI tracks of vehicle (splitted in query and test). 

``` 
python3 run_example.py
```








