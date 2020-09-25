import argparse
import torch
import torchvision
import json


class densenet_veri(torch.nn.Module):
    def __init__(self, old_model, nb_classes):
        super(densenet_veri, self).__init__()
        self.features = old_model.features
        self.mixer = torch.nn.AvgPool2d(kernel_size=7, stride=1)
        self.classifier = torch.nn.Linear(1920, nb_classes)

    def forward(self, x):
        x = self.features(x)
        x = self.mixer(x)
        f = x.view(x.size(0), -1)
        x = self.classifier(f)
        return f, x

def compute(dataset_dir, weights):
    """extract_feat : Extract Latent Representation (LR) of tracks (set of images)
        from the latent space (second-to-last layer) of the DenseNet201 CNN architecture
        (fine-tuned on the VeRI dataset for vehicle classification).

    Args:
        dataset_dir (str): Directory (path) of the image dataset
        weights (str): Filepath of the weights of the CNN 

    Returns:
        [type]: [description]
    """

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    data_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize(256),
        torchvision.transforms.CenterCrop(224),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    batch_size=1
    image_dataset = torchvision.datasets.ImageFolder(dataset_dir, data_transform)
    dataloader = torch.utils.data.DataLoader(
        image_dataset, batch_size=batch_size, shuffle=False, num_workers=1)
    dataset_size = len(image_dataset)
    class_names = image_dataset.classes

    model = torchvision.models.densenet201(pretrained=False)
    model = densenet_veri(model, 576) # modified version of the DenseNet201 CNN for VeRI

    if weights:
        if device.type == 'cuda':
            model.load_state_dict(torch.load(
                weights, map_location='cuda'))
        else:
            model.load_state_dict(torch.load(weights, map_location='cpu'))

    model = model.to(device)

    model.eval()

    features = {}
    n_t = 0
    for img, label in dataloader:
        img = img.to(device)
        feat, _ = model(img)

        for i, f in enumerate(feat):
            class_img = class_names[label[i].data]
            if class_img in features:
                if device.type == 'cuda':
                    features[class_img].append(
                        f.cpu().detach().numpy().tolist())
                else:
                    features[class_img].append(f.detach().numpy().tolist())
            else:
                features[class_img] = []
                if device.type == 'cuda':
                    features[class_img].append(
                        f.cpu().detach().numpy().tolist())
                else:
                    features[class_img].append(f.detach().numpy().tolist())
            n_t += 1
            print(f"completion : {(100*n_t/dataset_size):.2f}% ({n_t}/{dataset_size}) - current track : {class_img}", end="\r")
    print()
    return features

def main(args):
    features = compute(args.dataset_dir, args.weights)
    with open(args.output_file, "w") as f:
        f.write(json.dumps(features))

if __name__ == '__main__':
    parser=argparse.ArgumentParser(
        description = """Extract Latent Representation (LR) of tracks (set of images)
        from the latent space (second-to-last layer) of the DenseNet201 CNN architecture
        (fine-tuned on the VeRI dataset for vehicle classification).
        """)
    parser.add_argument("-d", "--dataset_dir",
                        dest = "dataset_dir",
                        type = str,
                        default = "./data/image_sample/query_tracks/",
                        help = "Dataset directory",)
    parser.add_argument("-o", "--output_file",
                        dest = "output_file",
                        type = str,
                        default = "data/results/feat_query.json",
                        help = "Output file",)
    parser.add_argument("-w", "--weights",
                        dest = "weights",
                        type = str,
                        default = "data/cnn_weights/VeRI_densenet_ft50.pth",
                        help = "File of weights of the Densenet201 model")
    args=parser.parse_args()
    main(args)
