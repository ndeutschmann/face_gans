import torch
import torchvision
import os
from src.models.dcgan32 import DCGen32


def generate_dcgan32_inversion_dataset(dataset_root="data/processed/dcgan32_inversion",
                                       dataset_size=100000,
                                       batch_size=128,
                                       generator_checkpoint_path="models/dcgan32v1/model_weights/checkpointG.2020_04_26",
                                       device=torch.device("cuda:0"),
                                       torch_seed = None):

    generator = DCGen32().to(device)
    generator_checkpoint = torch.load(generator_checkpoint_path)
    generator.load_state_dict(generator_checkpoint["model_state_dict"])
    generator.eval()
    print("Successfully initialized generator")

    imageconverter = torchvision.transforms.Compose([
        torchvision.transforms.Normalize((-1, -1, -1), (2, 2, 2)),
        torchvision.transforms.ToPILImage()
    ])

    image_dir = "imgs"
    vector_dir = "vecs"

    os.makedirs(dataset_root, exist_ok=True)
    os.makedirs(os.path.join(dataset_root, image_dir), exist_ok=False)
    os.makedirs(os.path.join(dataset_root, vector_dir), exist_ok=False)

    if torch_seed is not None:
        torch.manual_seed(torch_seed)

    n_images = 0
    while dataset_size > n_images:
        x = torch.zeros(batch_size, 100).normal_(0, 1.).to(device)
        batch = generator(x).cpu()
        x = x.cpu()

        for i, b in enumerate(batch):
            imageconverter(b).save(os.path.join(dataset_root, image_dir, "{}.png".format(i + n_images)))
            torch.save(x[i], os.path.join(dataset_root, vector_dir, "{}.pkl".format(i + n_images)))
        n_images += batch_size
        print("Generated and saved: {}/{} | {}".format(n_images,dataset_size,"="*(10*n_images//dataset_size)))

    if n_images < dataset_size:
        x = torch.zeros(dataset_size-n_images, 100).normal_(0, 1.).to(device)
        batch = generator(x).cpu()
        x = x.cpu()

        for i, b in enumerate(batch):
            imageconverter(b).save(os.path.join(dataset_root, image_dir, "{}.png".format(i + n_images)))
            torch.save(x[i], os.path.join(dataset_root, vector_dir, "{}.pkl".format(i + n_images)))