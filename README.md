# Vehicle Re-Identification using CNN Latent Spaces 

This repository is related to the publication "Vehicle Re-Identification using CNN Latent Spaces" (https://arxiv.org/abs/1910.09458). 

**Abstract**
```
This paper addresses the problem of vehicle re-identification using distance comparison of images in CNN latent spaces. First, we study the impact of the distance metrics, comparing performances obtained with different metrics: the minimal Euclidean distance (MED), the minimal cosine distance (MCD), and the residue of the sparse coding reconstruction (RSCR). These metrics are applied using features extracted through five different CNN architectures, namely ResNet18, AlexNet, VGG16, InceptionV3 and DenseNet201. We use the specific vehicle re-identification dataset VeRI to fine-tune these CNNs and evaluate results.
In overall, independently from the CNN used, MCD outperforms MED, commonly used in the literature. Secondly, the state-of-the-art image-to-track process (I2TP) is extended to a track-to-track process (T2TP) without using complementary metadata. Metrics are extended to measure distance between tracks, enabling the evaluation of T2TP and comparison with I2TP using the same CNN models. Results show that T2TP outperforms I2TP for MCD and RSCR. T2TP combining DenseNet201 and MCD-based metrics exhibits the best performances, outperforming the state-of-the-art I2TP models that use complementary metadata.
Finally, our experiments highlight two main results: i) the importance of the metric choice for vehicle re-identification, and ii) T2TP improves the performances compared to I2TP, especially when coupled with MCD-based metrics.
```

It contains the various python modules to 
1. Extract the latent representation of images of vehicle using the second-to-last layer of our CNN fine-tuned in the task of vehicle recognition as proposed in our paper. The CNN considered here is based on the DenseNet201 (https://arxiv.org/abs/1608.06993) architectures which has been fine-tuned using the VeRI dataset (https://github.com/VehicleReId/VeRidataset). 
2. Compute the Ranking Vehicle Re-identification between tracks of vehicle using the track-to-track metric ```meanMCD50``` proposed in our paper. 




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







