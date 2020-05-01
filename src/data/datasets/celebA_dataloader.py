import torch
import torchvision


def create_celebA_dataloader(image_size,batch_size=128,num_workers=1,data_root="data/processed"):

    my_transform=torchvision.transforms.Compose([
                                   torchvision.transforms.Resize(image_size),
                                   torchvision.transforms.CenterCrop(image_size),
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ])

    dataset = torchvision.datasets.ImageFolder(root=data_root,
                               transform=my_transform)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                             shuffle=True, num_workers=num_workers)

    return dataloader