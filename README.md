# Vehicle Re-Identification using CNN Latent Spaces 

This repository is related to the publication "Vehicle Re-Identification using CNN Latent Spaces" (https://arxiv.org/abs/1910.09458). 

The current repository contained the main python applications to reproduce Vehicle Re-identification ranking proposed in this original publication.

The CNN considered here is based on the DenseNet201 (https://arxiv.org/abs/1608.06993) architectures which has been fine-tuned using the VeRI dataset (https://github.com/VehicleReId/VeRidataset). 


## Main files
* ReID_ModelVeRI_ExtractFeat.py
    - extract LR of each track of vehicle (Query & Test) -> return a json files containing the features for each track
* ReID_ModelVeRI_Ranking.py
    - compute the ranking for each track of the Query database -> return a json file containing the ranking for each query track
* VeRI_densenet_ft50.pth
    - Pre-trained weights for the DenseNet201 architecture. The model has been trained to classify vehicles of training set the VeRI dataset. Only its latent space (the second-to-last layer) is used to extract features

## Dependencies
* Numpy
* PIL
* Pytorch
* torchvision
* Matplotlib







