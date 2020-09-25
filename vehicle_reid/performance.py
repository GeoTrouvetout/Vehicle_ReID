import json
import argparse
import numpy as np

def compute(ranking):
    """[summary]

    Args:
        ranking (dict): Python dict containing the ranking of each query track

    Returns:
        mAP (float):  See more details in the original paper (https://arxiv.org/abs/1910.09458)
        rank1 (float): Proportion of successful vehicle retrieval in the top 1 position of the rank. Equivalent to accuracy
        rank5 (float): Proportion of successful vehicle retrieval in the top 5 positions of the rank.
    """

    rank1 = 0
    rank5 = 0
    mAP = 0
    n = 0
    for _, ranking_track in ranking["tracks"].items():
        id_class = ranking_track["class"]
        list_class_rankink = [
                ranking_track["ranking"][str(i+1)]["class"] 
                for i in range(len(ranking_track["ranking"])) 
            ]
        if ranking_track["class"] == list_class_rankink[0] : 
            rank1 += 1
        if ranking_track["class"] in list_class_rankink[0:5] : 
            rank5 += 1
        Nq = sum([id_class == _class for _class in list_class_rankink])

        AP= sum([ 
                    sum( 
                        [ id_class == ranking_track["ranking"][str(i+1)]["class"]  for i in np.arange(0, r+1)]
                    )*
                    (
                         id_class == ranking_track["ranking"][str(r+1)]["class"] 
                    )/(r+1) 
                    for r in range(len(ranking_track["ranking"])) if id_class == ranking_track["ranking"][str(r+1)]["class"] 
                ])
        AP /= Nq
        mAP += AP
        n += 1
    mAP /= n
    rank1 /= n
    rank5 /= n
    
    return mAP, rank1, rank5

def main(args):
    if args.ranking:
        with open(args.ranking, 'r') as f:
            d_ranking = json.load(f)
    mAP, rank1, rank5 = compute(d_ranking)

    print("mAP : ", mAP)
    print("rank1 : ", rank1)
    print("rank5 : ", rank5)



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Compute ranking and overall performance metrics.")
    parser.add_argument("-r", "--ranking",
                        dest="ranking",
                        type=str,
                        default = "data/results/ranking.json",
                        help="Input file containing the ranking for each query track (Need file format returned by ReID_ModelVeRI_ComputeDistance.py)",)  
    args = parser.parse_args()
    main(args)






