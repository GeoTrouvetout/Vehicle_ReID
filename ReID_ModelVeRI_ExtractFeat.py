from __future__ import print_function, division

from PIL import Image
import argparse

import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

import json

import torch.nn.functional as F

from torch.utils.data.sampler import Sampler


class densenet_veri(nn.Module):
    def __init__(self, old_model, nb_classes):
        super(densenet_veri, self).__init__()
        self.features = old_model.features
        self.mixer =  torch.nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = nn.Linear(1920, nb_classes) 
    def forward(self, x):
        x = self.features(x)
        x = self.mixer(x)
        f = x.view(x.size(0), -1)
        x = self.classifier(f)
        return f, x

class NumpyEncoder(json.JSONEncoder):
    """ Special json encoder for numpy types """
    def default(self, obj):
        if isinstance(obj, (np.int_, np.intc, np.intp, np.int8,
            np.int16, np.int32, np.int64, np.uint8,
            np.uint16, np.uint32, np.uint64)):
            return int(obj)
        elif isinstance(obj, (np.float_, np.float16, np.float32, 
            np.float64)):
            return float(obj)
        elif isinstance(obj,(np.ndarray,)): #### This is the fix
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)

def main(args):
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image_dataset = datasets.ImageFolder(args.datadir,data_transform)
    dataloader =  torch.utils.data.DataLoader(image_dataset, batch_size=args.batchsize, shuffle=False, num_workers=1)
    dataset_size = len(image_dataset)
    class_names = image_dataset.classes
    n_cuda=args.ncuda
    device = torch.device("cuda:"+str(n_cuda) if torch.cuda.is_available() else "cpu")


    model = models.densenet201(pretrained=True)
    model = densenet_veri(model, 576)

    print(model.eval(), device.type)

    if args.weight:
        if device.type == 'cuda':
            model.load_state_dict(torch.load(args.weight))
        else:
            #torch.load('my_file.pt', map_location=lambda storage, location: 'cpu')
            model.load_state_dict(torch.load(args.weight, map_location='cpu'))

    model = model.to(device)

    model.eval()

    running_data = 0.0

    features = {}
    n_t = 0
    for img, label in dataloader:
        img = img.to(device)
        feat, output = model(img)
        if n_t > 9:
            break
        for i, f in enumerate(feat):
            class_img = class_names[label[i].data]    
        
            if class_img in features:
                if device.type == 'cuda':
                    features[class_img].append(f.cpu().detach().numpy().tolist())
                else:
                    features[class_img].append(f.detach().numpy().tolist())
            else:
                features[class_img] = []
                if device.type == 'cuda':
                    features[class_img].append(f.cpu().detach().numpy().tolist())
                else:
                    features[class_img].append(f.detach().numpy().tolist())
            n_t += 1
            print(class_img, "{:.2f}%".format(100*n_t/dataset_size), n_t, dataset_size, end="\r")
        
    print()
 
    
    with open(args.outfile,"w") as tmpfile:
        tmpfile.write(json.dumps(features))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Program to extract Latent Representation from the latent space of the CNN Densenet201.")
    parser.add_argument("-d", "--dataset_dir",
                        dest="datadir",
                        type=str,
                        default="./",
                        help="Dataset directory",)  
    parser.add_argument("-o", "--output_file",
                        dest="outfile",
                        type=str,
                        default="output.json",
                        help="Output file",)  
    parser.add_argument("-w", "--weight",
                        dest="weight",
                        type=str,
                        help="weight of the Densenet201 model")
    parser.add_argument("-b", "--batch_size",
                        dest="batchsize",
                        type=int,
                        default=1,
                        help="batch size. Default 1")
    parser.add_argument("-c", "--cuda",
                        dest="ncuda",
                        type=int,
                        default=0,
                        help="ID of the CUDA device to use (0 by default).(CPU if no CUDA is available). ")
    args = parser.parse_args()
    main(args)






