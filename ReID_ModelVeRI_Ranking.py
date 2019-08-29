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
    device = torch.device("cuda:"+str(n_cuda) if torch.cuda.is_available() else "cpu")
    cos= nn.CosineSimilarity(dim=1, eps=1e-6)

    running_corrects = 0.0

    running_data = 0.0

    ### Load LR for every track of the DB
    if args.testfeat:
        with open(args.testfeat, 'r') as f:
            features_test = json.load(f)
    
    if args.queryfeat:
        with open(args.queryfeat, 'r') as f:
            features_query = json.load(f)

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
        MATCH[cquery] = {}
        
        for i, (d, c) in enumerate(sorted(zip(listd, listc))):
            MATCH[cquery][i+1] = {"id": c, "distance":d}
        

    #### SAVE
    with open(args.outfile,"w") as tmpfile:
        tmpfile.write(json.dumps(MATCH))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Script to compute the ranking of tracks.")
    parser.add_argument("-q", "--query_feat",
                        dest="queryfeat",
                        type=str,
                        help="Query feat file",)  
    parser.add_argument("-t", "--test_feat",
                        dest="testfeat",
                        type=str,
                        help="Test feat file",)  
    parser.add_argument("-o", "--output_file",
                        dest="outfile",
                        type=str,
                        default="ranking.json",
                        help="Output rank file",)  
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






