import os
import torch
import torchvision
from ..datasets import ImageTargetFoldersDataset,ImageFoldersTargetFileDataset,MultiHDF5TablesDataset

pil_image_transform = torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])


def create_dcgan32_inversion_dataloader_many_files(root="data/processed/dcgan32_inversion",
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
                                             num_workers=num_workers,
                                             **kwargs)

    return dataloader

def create_dcgan32_inversion_dataloader_many_images(root="data/processed/dcgan32_inversion",
                                                    image_dir="imgs",
                                                    target_path="vecs/labels.pkl",
                                                    transform=pil_image_transform,
                                                    batch_size=128,
                                                    num_workers=2,
                                                    **kwargs
                                                   ):

    dataset = ImageFoldersTargetFileDataset(root=root,
                                            image_dir=image_dir,
                                            target_path=target_path,
                                            transform=transform,
                                            )

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             **kwargs)

    return dataloader

def create_dcgan32_inversion_dataloader_hdf5_tables(root="data/processed/dcgan32_inversion",
                                                    image_file="images.h5",
                                                    label_file="labels.h5",
                                                    image_node="/data/images",
                                                    label_node="/data/labels",
                                                    batch_size=128,
                                                    num_workers=2,
                                                    **kwargs
                                                   ):
    files= os.path.join(root, image_file), os.path.join(root, label_file)
    arrays = image_node, label_node
    dataset = MultiHDF5TablesDataset(files,
                                     arrays,
                                     load_in_memory=(False,True)
                                     )

    dataloader = torch.utils.data.DataLoader(dataset,
                                             batch_size=batch_size,
                                             shuffle=True,
                                             num_workers=num_workers,
                                             **kwargs)

    return dataloader