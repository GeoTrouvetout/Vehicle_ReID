# Vehicle Re-Identification using Track-to-track ranking of deep latent representation of vehicles

This repository is related to the publication "Improving Vehicle Re-Identification using CNN Latent Spaces: Metrics Comparison and Track-to-track Extension" (https://arxiv.org/abs/1910.09458). This paper is a postprint of the paper submitted and accepted to IET Computer Vision (https://digital-library.theiet.org/content/journals/iet-cvi).


The python module ```vehicle_reid``` allow you to 
1. Extract the latent representation of images of vehicle using the second-to-last layer of our CNN fine-tuned in the task of vehicle recognition as proposed in our paper. The CNN considered here is based on the DenseNet201 (https://arxiv.org/abs/1608.06993) architectures which has been fine-tuned using the VeRI dataset (https://github.com/VehicleReId/VeRidataset). Corresponding weights are given in ```"data/cnn_weights/VeRI_densenet_ft50.pth"```
2. Compute the Ranking Vehicle Re-identification between tracks of vehicle using the various distance metrics studied in the paper. 
3. Compute the performance metrics rank1 rank5 and mAP.


## Dependencies
- numpy==1.19.2
- torchvision==0.7.0
- torch==1.6.0
- scikit_learn==0.23.2


## The package ```vehicle_reid```
The module ```vehicle_reid``` is composed of 3 independent modules 
- ```latent_representation.py```
    - extract LR of each track of vehicle -> return a json file containing the LR for each track
- ```ranking.py```
    - compute the ranking for each query track -> return a json file containing the ranking for each query track
- ```performance.py```
    - compute the performance metrics. namely rank1 rank5 and mAP (See paper for details)



## Running example 
The directory ```data``` contains data to test the module ```vehicle_reid```. Note that to perform the VeRI experiments presented on the paper, you'll need the VeRI dataset which can be found, by simple request to authors, here : https://github.com/JDAI-CV/VeRidataset

- data/cnn_weights/VeRI_densenet_ft50.pth : Pre-trained weights for the DenseNet201 architecture. The model has been trained to classify vehicles of training set the VeRI dataset. Only its latent space (the second-to-last layer) is used to extract features
- data/image_sample : some VeRI tracks of vehicle (splitted in query and test). 

``` 
python3 run_example.py
```








