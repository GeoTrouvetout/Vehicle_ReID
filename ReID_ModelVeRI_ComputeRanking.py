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
    image_dataset = datasets.ImageFolder(args.datadir,data_transform)
    dataloader =  torch.utils.data.DataLoader(image_dataset, batch_size=args.batchsize, shuffle=False, num_workers=1)
    dataset_size = len(image_dataset)
    class_names = image_dataset.classes
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
    if args.dbjson:
        with open(args.dbjson, 'r') as f:
            features_db = json.load(f)
    
    ### compute LR for every query
    features_query = {}
    n_t = 0
    for query, lquery in dataloader:
        query = query.to(device)
        featq, outq = model(query)
        
        #if n_t == 10:
        #    break
        for i, f in enumerate(featq):
            cquery = class_names[lquery[i].data]    
            n_t += 1

            print(cquery, "{:.2f}%".format(100*n_t/dataset_size), n_t, dataset_size, end="\r")
        
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
        n_ctest = len(features_db)
        for ctest, ftest in features_db.items():
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
                    #d = torch.norm(g - f)
                    d = 1 - cos(g.unsqueeze(0), f.unsqueeze(0))
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
    parser = argparse.ArgumentParser(description="Script to compute the ranking of tracks for every query track.")
    parser.add_argument("-d", "--dataset_dir",
                        dest="datadir",
                        type=str,
                        default="./",
                        help="Query tracks directory",)  
    parser.add_argument("-i", "--db_feat",
                        dest="dbjson",
                        type=str,
                        help="DB feat file",)  
    parser.add_argument("-o", "--output_file",
                        dest="outfile",
                        type=str,
                        default="output.json",
                        help="Output rank file",)  
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






