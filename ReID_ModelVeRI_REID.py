from __future__ import print_function, division

import sys
import os
import json
import argparse


import torch
import torch.nn as nn

import numpy as np
import torchvision
from torchvision import datasets, models, transforms

from torch.utils.data.sampler import Sampler


def match_to_cmc(match):
    cmc = {}
    n_query = 0
    for cquery, value in match.items():
        n_query += 1.0
        listd = value["dist"]
        listc = value["class"]
        listc_sort = [c for _, c in sorted(zip(listd, listc))]
        for i in range(len(listc_sort)):
            if not i in cmc:
                cmc[i] = 0
            if cquery.split('_')[0] in [c_.split('_')[0] for c_ in listc_sort[:i+1]]:
                cmc[i]+=1

    cmc['N'] = n_query
    return cmc
 


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


def main(args):
    data_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    image_dataset_query = datasets.ImageFolder(args.querydir,data_transform) 
    dataloader_query =  torch.utils.data.DataLoader(image_dataset_query, batch_size=args.batchsize, shuffle=False, num_workers=1)
    dataset_size_query = len(image_dataset_query)
    class_names_query = image_dataset_query.classes

    image_dataset_test = datasets.ImageFolder(args.testdir,data_transform) 
    dataloader_test =  torch.utils.data.DataLoader(image_dataset_test, batch_size=args.batchsize, shuffle=False, num_workers=1)
    dataset_size_test = len(image_dataset_test)
    class_names_test = image_dataset_test.classes

    n_cuda=args.ncuda
    device = torch.device("cuda:"+str(n_cuda) if torch.cuda.is_available() else "cpu")

    model = models.densenet201(pretrained=True)
    model = densenet_veri(model, 576)

    print(model.eval())


    model = model.to(device)

    if args.weight:
        if device.type == 'cuda':
            model.load_state_dict(torch.load(args.weight))
        else:
            #torch.load('my_file.pt', map_location=lambda storage, location: 'cpu')
            model.load_state_dict(torch.load(args.weight, map_location='cpu'))

    cos= nn.CosineSimilarity(dim=1, eps=1e-6)

    model.eval()   # Set model to evaluate mode

    running_corrects = 0.0

    running_data = 0.0

    ### Load LR for every track of the DB
    features_test = {}
    n_t = 0
    for test, ltest in dataloader_test:
        test = test.to(device)
        featt, outt = model(test)
        
        #if n_t == 10: # for debugging
        #    break
        for i, f in enumerate(featt):
            ctest = class_names_test[ltest[i].data]    
            n_t += 1

            print(ctest, "{:.2f}%".format(100*n_t/dataset_size_test), n_t, dataset_size_test, end="\r")
        
            if ctest in features_test:
                if device.type == 'cuda':
                    features_test[ctest].append(f.cpu().detach().numpy())
                else:
                    features_test[ctest].append(f.detach().numpy())
            else:
                features_test[ctest] = []
                if device.type == 'cuda':
                    features_test[ctest].append(f.cpu().detach().numpy())
                else:
                    features_test[ctest].append(f.detach().numpy())
    
    ### compute LR for every query
    features_query = {}
    n_t = 0
    for query, lquery in dataloader_query:
        query = query.to(device)
        featq, outq = model(query)
        
        #if n_t == 10: # for debugging
        #    break
        for i, f in enumerate(featq):
            cquery = class_names_query[lquery[i].data]    
            n_t += 1

            print(cquery, "{:.2f}%".format(100*n_t/dataset_size_query), n_t, dataset_size_query, end="\r")
        
            if cquery in features_query:
                if device.type == 'cuda':
                    features_query[cquery].append(f.cpu().detach().numpy())
                else:
                    features_query[cquery].append(f.detach().numpy())
            else:
                features_query[cquery] = []
                if device.type == 'cuda':
                    features_query[cquery].append(f.cpu().detach().numpy())
                else:
                    features_query[cquery].append(f.detach().numpy())


    MATCH = {}

    for cquery, fquery in features_query.items():
        
        if device.type == 'cuda':
            fquery = torch.FloatTensor(fquery).cuda(n_cuda)
        else:
            fquery = torch.FloatTensor(fquery)
        running_data += 1.0        

        print(fquery.shape)

        listc =[]
        listd = []
        dmin = 1000000000
        class_found=""

        n_t=0
        n_ctest = len(features_test)
        for ctest, ftest in features_test.items():
            if device.type == 'cuda':
                ftest = torch.FloatTensor(ftest).cuda(n_cuda)
            else:
                ftest = torch.FloatTensor(ftest)
            n_t += 1
            listc.append(ctest)
            l_d = []
            for f in ftest: 

                _dmin = 1000000000
                for g in fquery:
                    if args.metric == "MCD":
                    #d = torch.norm(g - f)
                        d = 1 - cos(g.unsqueeze(0), f.unsqueeze(0))
                    elif args.metric == "MED":
                        d = torch.norm(g - f)
                    else:
                        raise Exception('metric should be either MED or MCD')
                    _dmin = min(_dmin, d.item() )
                    l_d.append(_dmin)
                l50_d = sorted(l_d)[:int(len(l_d)/2)+1]
                
                d = np.mean(l50_d)
                listd.append(d)
                if d < dmin:
                    dmin = d
                    class_found = ctest
        print(listd)
        print(listc)
        MATCH[cquery] = {}
        MATCH[cquery]["dist"] = listd
        MATCH[cquery]["class"] = listc


    #### SAVE
    with open(args.outfile,"w") as tmpfile:
        tmpfile.write(json.dumps(MATCH))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script for the re-identication process. Each Query track is compare to every Test tracks using Latent representation of their images (using the CNN DenseNet201). The comparison is done using a distance metric based on the Minimum-Euclidean-Distance (MED) or Minimum-Cosine-Distance (MCD). MCD is set by default. In case of track-to-track comparison, the aggregation function is defined as the mean of the second quartile (the first half) the image-to-image distances (mean50MCD, or MEAN50MED). Then, a ranking is constructed, from the most similar Test tracks to the last.")
    parser.add_argument("-t", "--test_dir",
                        dest="testdir",
                        type=str,
                        default="./",
                        help="Test tracks directory",)  
    parser.add_argument("-q", "--query_dir",
                        dest="querydir",
                        type=str,
                        default="./",
                        help="Query tracks directory",)  
    parser.add_argument("-o", "--output_file",
                        dest="outfile",
                        type=str,
                        default="ranking.json",
                        help="Output ranking file",)  
    parser.add_argument("-w", "--weight",
                        dest="weight",
                        type=str,
                        help="weight of the Densenet201 model")
    parser.add_argument("-m", "--metric",
                        dest="metric",
                        type=str,
                        default="MCD",
                        help="Distance metric used. MED or MCD (Default: MCD). For track-to-track distance computation, the aggregation function is mean50 (the mean of the smallest image-to-image distances)",)  
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






