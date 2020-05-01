import torch
import torchvision
from ..datasets import ImageTargetFoldersDataset

pil_image_transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])


def create_dcgan32_inversion_dataloader(root="data/processed/dcgan32_inversion",
                                        image_dir="imgs",
                                        target_dir="vecs",
                                        target_ext="pkl",
                                        transform=pil_image_transform,
                                        batch_size=128,
                                        num_workers=2,
                                        **kwargs
                                        ):

    dataset = ImageTargetFoldersDataset(root=root,
                                        image_dir=image_dir,
                                        target_dir=target_dir,
                                        target_ext=target_ext,
                                        transform=transform)

    dataloader = torch.utils.data.DataLoader(dataset,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=num_workers,**kwargs)

    return dataloader