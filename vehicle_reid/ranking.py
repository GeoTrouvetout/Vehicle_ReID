from __future__ import print_function, division

import sys
import os
import json
import argparse

import torch

import numpy as np
import torchvision


from sklearn.decomposition import SparseCoder



def compute_metric(query, test, metric="MCD"):
    """compute_metric Compute distance between two vectors [query] and [test]. 

    Args:
        query (torch.Tensor()): LR vector of the query 
        test (torch.Tensor()): LR vector of the test
        metric (str, optional): Distance metric to use : MED for euclidean distance, MCD for cosine distance. Defaults to "MCD".

    Raises:
        Exception: metric must be either MED, MCD.

    Returns:
        float: distance between the two vectors
    """
    if metric == "MCD":
        cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
        return 1 - cos(query.unsqueeze(0), test.unsqueeze(0)).item()
    elif metric == "MED":
        return torch.norm(query - test).item()
    else:
        raise Exception(
            "Metric function Error : metric must be either MED, MCD.")


def compute_aggregation(distances, aggregation="mean50"):
    """compute_aggregation [summary]

    Args:
        distances (list): List containing the distance (MED or MCD) between LR
        aggregation (str, optional): Aggregation function. Defaults to "mean50".

    Returns:
        float: Aggregated distance
    """
    if "50" in aggregation:
        distances = sorted(distances)[:int(len(distances)/2)+1]
    if "mean" in aggregation:
        d = np.mean(distances)
    elif "med" in aggregation:
        d = np.median(distances)
    elif "min" in aggregation:
        d = np.min(distances)
    else:
        raise Exception(
            "Aggregation function Error : Aggregation must be either min, mean, med, mean50 or mean50.")
    return d


def compute_residual(query, test):
    """Compute the residual of the sparse coding representation (RSCR) 

    Args:
        query (torch.Tensor): Query LR
        test (torch.Tensor): Test LR

    Returns:
        float: RSCR
    """

    D = test.squeeze().detach().numpy()
    Y = query.detach().numpy()
    
    X = query
    
    if Y.ndim < 2:
        Y = Y.reshape(1, -1)
    if D.ndim < 2:
        D = D.reshape(1, -1)
    coder = SparseCoder(dictionary=D, transform_algorithm='lasso_lars')
    A = coder.transform(Y)
    A = torch.FloatTensor(A) 
    D = torch.FloatTensor(D)
    RSCR = torch.norm(X - torch.mm(A, D)).item()

    return RSCR

def compute(features_query, features_test, metric="MCD", aggregation="mean50"):
    """Compute the ranking of test tracks for each query track according to a distance metric

    Args:
        features_query (dict): contains LR of the query tracks
        features_test (dict): contains LR of the test tracks
        metric (str, optional): Distance Metric to use (MED|MCD). Defaults to "MCD".
        aggregation (str, optional): Aggregation function (min|mean|med|mean5). Defaults to "mean50".

    Returns:
        dict: contains ranking for each of the query track
    """

    ranking = {"metric": metric, "aggregation": aggregation, "tracks" : {}}

    print()

    len_query = len(features_query)
    n_query = 0
    for cquery, fquery in features_query.items():
        n_query += 1
        listc =[]
        listd = []

        fquery = torch.FloatTensor(fquery)

        # Note : track_id is in format 'class_camera'
        ranking["tracks"][cquery] = {"class" : cquery.split("_")[0], "ranking" : {}} 

        len_test = len(features_test)
        n_test = 0

        for ctest, ftest in features_test.items():
            n_test += 1
            ftest = torch.FloatTensor(ftest)

            if metric == "RSCR":
                d = compute_residual(fquery, ftest)
            else:
                l_d = []

                for t in ftest:

                    _dmin = np.inf # initialization for min comparison with distance
                    for q in fquery:
                        d = compute_metric(q, t, metric=metric)
                        _dmin = min(_dmin, d)
                        l_d.append(_dmin)

                d = compute_aggregation(l_d, aggregation)
                
            print(f"completion global : {(100*n_query/len_query):.2f}% ({n_query}/{len_query}) - current track : {(100*n_test/len_test):.2f}% ({n_test}/{len_test})", end="\r")
            listd.append(d)
            listc.append(ctest)
            for i, (d, c) in enumerate(sorted(zip(listd, listc))):
                ranking["tracks"][cquery]["ranking"][str(i+1)] = { "id": c, "class" : c.split('_')[0], "distance":d}
    print()
    

    return ranking

def main(args):
 
    # Load LRs for query and tracks
    if args.queryfeat:
        with open(args.queryfeat, 'r') as f:
            features_query = json.load(f)

    if args.testfeat:
        with open(args.testfeat, 'r') as f:
            features_test = json.load(f)

    d_ranking = compute(features_query, 
                        features_test, 
                        metric=args.metric, 
                        aggregation=args.aggregation)

    with open(args.output_file, "w") as f:
        json.dump(d_ranking, f, indent=4)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Compute the ranking of test tracks for each query track according to a distance metric.")
    parser.add_argument("-q", "--query_feat",
                        dest="queryfeat",
                        type=str,
                        default="data/results/feat_query.json",
                        help="Query feat file",)
    parser.add_argument("-t", "--test_feat",
                        dest="testfeat",
                        type=str,
                        default="data/results/feat_test.json",
                        help="Test feat file",)
    parser.add_argument("-o", "--output_file",
                        dest="output_file",
                        type=str,
                        default="data/results/ranking.json",
                        help="Output rank file",)
    parser.add_argument("-m", "--metric",
                        dest="metric",
                        type=str,
                        default="MCD",
                        help="Distance metric to use. MED MCD or RSCR (Default: MCD)")
    parser.add_argument("-a", "--aggregation",
                        dest="aggregation",
                        default="mean50",
                        help="""For track-to-track (T2T) only.  \n
                        Ignored if image-to-track (I2T : query is one image and test is a track) or if metric = \"RSCR\". 
                        Choices are : min, mean,  med or mean50 (Default : mean50) \n
                        See original paper for details about the aggregation function in T2T 
                        (https://arxiv.org/abs/1910.09458)""",
                        )
    args = parser.parse_args()
    main(args)
