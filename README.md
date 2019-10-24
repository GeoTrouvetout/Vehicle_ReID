# Vehicle Re-Identification using CNN Latent Spaces 

This repository is related to the publication "Vehicle Re-Identification using CNN Latent Spaces" (https://arxiv.org/abs/1910.09458). 
The current repository contained the main python applications to reproduce Vehicle Re-identification ranking proposed in this original publication.

## Main files
* ReID_ModelVeRI_ExtractFeat.py
  - extract LR of each track of vehicle (Query & Test) -> return a json files containing the features for each track
* ReID_ModelVeRI_Ranking.py
* VeRI_densenet_ft50.pth

The CNN considered here is based on the DenseNet201 (https://arxiv.org/abs/1608.06993) architectures which has been fine-tuned using the VeRI dataset (https://github.com/VehicleReId/VeRidataset). 


## Dependencies
* Numpy
* PIL
* Pytorch
* torchvision
* Matplotlib







