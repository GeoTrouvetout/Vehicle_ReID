import json
import vehicle_reid
from vehicle_reid import latent_representation, ranking, performance


dir_query = "data/image_sample/query_tracks"
dir_test = "data/image_sample/test_tracks"
file_weights = "data/cnn_weights/VeRI_densenet_ft50.pth"

lr_query = latent_representation.compute(dataset_dir = dir_query, weights=file_weights)
with open("data/results/feat_query.json", "w") as f:
    f.write(json.dumps(lr_query))

lr_test = latent_representation.compute(dataset_dir = dir_test, weights=file_weights)
with open("data/results/feat_test.json", "w") as f:
    f.write(json.dumps(lr_test))


d_ranking = ranking.compute(lr_query, lr_test, metric="RSCR")
with open("data/results/ranking.json", "w") as f:
    f.write(json.dumps(d_ranking))

mAP, rank1, rank5 = performance.compute(d_ranking)
print("RSCR")
print("mAP : ", mAP)
print("rank1 : ", rank1)
print("rank5 : ", rank5)
for m in ["MED", "MCD"]:
    for a in ["min", "mean", "med", "mean50", "med50"]:
        d_ranking = ranking.compute(lr_query, lr_test, metric=m, aggregation=a)
        mAP, rank1, rank5 = performance.compute(d_ranking)
        print(m, a)
        print("mAP : ", mAP)
        print("rank1 : ", rank1)
        print("rank5 : ", rank5)