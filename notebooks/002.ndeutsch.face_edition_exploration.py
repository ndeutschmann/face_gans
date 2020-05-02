# -*- coding: utf-8 -*-
# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:light
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.4.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# **This notebook was written while moving from 04b4716457b134d8d74b4ff5f626435e1b02dc85 to 1ad206d92fc300ea5843438f36fe903b3c0e2477**
#
# Not runnable in current versions

# +
import torch
import torchvision
from PIL import Image
ngpu = 2
from matplotlib import pyplot as plt
import numpy as np
import torchvision.utils as vutils
from tqdm.autonotebook import tqdm
import os
import src.data.make_dataset as md
from src.models.util import GaussianNoise,MemBatch

device = torch.device("cuda:0")




# +
class ConvG(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.deconv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(100,256,4,1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(256,256,3,1,1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2),          
            torch.nn.ConvTranspose2d(256,128,4,1),            
            torch.nn.BatchNorm2d(128),            
            torch.nn.LeakyReLU(0.2),            
            torch.nn.ConvTranspose2d(128,64,4,2),
            torch.nn.BatchNorm2d(64),            
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(64,64,3,2,1),
            torch.nn.BatchNorm2d(64),            
            torch.nn.LeakyReLU(0.2),   
            torch.nn.ConvTranspose2d(64,32,3,1,1),
            torch.nn.BatchNorm2d(32),            
            torch.nn.LeakyReLU(0.2),             
            torch.nn.ConvTranspose2d(32,3,2,1),
            torch.nn.Tanh()
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(100,100),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(100,100),
            torch.nn.BatchNorm1d(100)
        )
        
        
    def forward(self,x):
        z = (self.dense(x)+x).unsqueeze(-1).unsqueeze(-1)
        return self.deconv(z)
    
class ConvD(torch.nn.Module):
    
    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3,16,2,2),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(16,64,4,2), 
            torch.nn.BatchNorm2d(64),            
            torch.nn.LeakyReLU(0.2),         
            torch.nn.Conv2d(64,128,4,1),   
            torch.nn.BatchNorm2d(128),            
            torch.nn.LeakyReLU(0.2),          
            torch.nn.Conv2d(128,256,4,1),
            torch.nn.BatchNorm2d(256),            
            torch.nn.LeakyReLU(0.2),             
        )
        
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(256,1),
            torch.nn.Sigmoid()            
        )
        
        
    def forward(self,x):
        z = self.conv(x)
        return self.dense(z.view(-1,256))    
# -

gw=torch.load("../models/dcgan32v1/model_weights/checkpointG.2020_04_26")
dw=torch.load("../models/dcgan32v1/model_weights/checkpointD.2020_04_26")

G=ConvG().cuda()
G.load_state_dict(gw["model_state_dict"])

# +
x = torch.zeros(64,100,device=device).normal_(0,.7)
Batch= G(x)

plt.figure(figsize=(10,10))
plt.imshow(np.transpose(vutils.make_grid(Batch, padding=2, normalize=True).detach().cpu(),(1,2,0)))
plt.show()
# -

# # Generating a dataset of z,img

import os

# !rm -rf dcgan32_inversion_dataset

dataset_path = "dcgan32_inversion_dataset"
image_dir = "imgs"
vector_dir = "vecs"
os.makedirs(dataset_path,exist_ok=False)
os.makedirs(os.path.join(dataset_path,image_dir),exist_ok=False)
os.makedirs(os.path.join(dataset_path,vector_dir),exist_ok=False)

nbatch = 32
imageconverter = torchvision.transforms.Compose([
    torchvision.transforms.Normalize((-1, -1, -1), (2, 2, 2)),
    torchvision.transforms.ToPILImage()
])

# +
i0 = 32
torch.manual_seed(42)
x = torch.zeros(nbatch,100).normal_(0,1.).to(device)
batch = G(x).cpu()
x = x.cpu()

for i,b in enumerate(batch):
    imageconverter(b).save(os.path.join(dataset_path,image_dir,"{}.png".format(i+i0)))
    torch.save(x[i],os.path.join(dataset_path,vector_dir,"{}.pkl".format(i+i0)))                           
# -

imageconverter(batch[2])

b2=batch[2].permute(1,2,0).detach().numpy()
plt.imshow((b2+1)/2)

# !ls dcgan32_inversion_dataset/vecs/

# ## Dataset generation command

# We've put all of the above in a nice command as part of the src library

from src.data.dcgan32 import generate_dcgan32_inversion_dataset
import os

# This is supposed to be ran from the project root (ultimately with a make inversion_data command)
os.chdir('/home/ndeutsch/face_gans/face_gans')
print(os.getcwd())
generate_dcgan32_inversion_dataset(torch_seed=42)

from PIL import Image

# !ls data/processed/dcgan32_inversion/imgs/1.png

Image.open("data/processed/dcgan32_inversion/imgs/12.png")

# ## Testing our dataset importer class

from torchvision.datasets.vision import VisionDataset
from torchvision.datasets.folder import default_loader
from src.data.image_target_folder_dataset import ImageTargetFoldersDataset
from PIL import Image
os.chdir('/home/ndeutsch/face_gans/face_gans')

# +
my_transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ])

dataset = ImageTargetFoldersDataset("data/processed/dcgan32_inversion","imgs","vecs",transform=my_transform)
# -

dataset.images.index("/home/ndeutsch/face_gans/face_gans/data/processed/dcgan32_inversion/imgs/2.png")

im=dataset[22576][0]
plt.imshow(im.permute(1,2,0)*0.5+0.5)
plt.show()

from src.models.dcgan32 import DCGen32
os.chdir('/home/ndeutsch/face_gans/face_gans')
G=DCGen32(load_weights=True)
G.eval()
v=dataset[22576][1]
plt.imshow(G(v.unsqueeze(0)).detach()[0].permute(1,2,0)*0.5+0.5)

G(v.unsqueeze(0)).shape

pilim=Image.open("/home/ndeutsch/face_gans/face_gans/data/processed/dcgan32_inversion/imgs/2.png")

plt.imshow(pilim)

# # Testing the dataloader

import src.data.datasets as datasets
import os

os.chdir('/home/ndeutsch/face_gans/face_gans')
dataloader = datasets.create_dcgan32_inversion_dataloader()

batch=next(iter(dataloader))

plt.imshow(batch[0][0].permute(1,2,0)*0.5+0.5)



from src.models.dcgan32 import DCGen32
os.chdir('/home/ndeutsch/face_gans/face_gans')
G=DCGen32(load_weights=True)
G.eval()
plt.imshow(G(batch[1][0].unsqueeze(0)).detach()[0].permute(1,2,0)*0.5+0.5)

# # Prototyping an inverser network

os.chdir('/home/ndeutsch/face_gans/face_gans')
dataloader = datasets.create_dcgan32_inversion_dataloader()
batch=next(iter(dataloader))

batch[0][0].shape

x = batch[0][:1]
print(x.shape)
x=torch.nn.Conv2d(3,32,3,1,1)(x)
print(x.shape)
x=torch.nn.Conv2d(32,128,3,2,1)(x)
print(x.shape)
x=torch.nn.Conv2d(128,256,2,2,1)(x)
print(x.shape)
x=torch.nn.Conv2d(256,256,2,2)(x)
print(x.shape)
x=torch.nn.Conv2d(256,512,2,4)(x)
print(x.shape)
x=x.squeeze(-1).squeeze(-1)
print(x.shape)
x


class DCGAN32Inverter(torch.nn.Module):
    def __init__(self,channels=32):
        super(DCGAN32Inverter,self).__init__()
        
        self.conv = torch.nn.Sequential(
            torch.nn.BatchNorm2d(3),
            torch.nn.Conv2d(3,channels,3,1,1),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels,channels*2,3,2,1),
            torch.nn.BatchNorm2d(channels*2),
            torch.nn.ReLU(),            
            torch.nn.Conv2d(channels*2,channels*4,2,2,1),
            torch.nn.BatchNorm2d(channels*4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels*4,channels*8,2,2),
            torch.nn.BatchNorm2d(channels*8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels*8,channels*16,2,4),
            torch.nn.BatchNorm2d(channels*16),
            torch.nn.ReLU()
        )
        
        self.dense = torch.nn.Sequential(            
            torch.nn.Linear(16*channels,16*channels),
            torch.nn.BatchNorm1d(channels*16),            
            torch.nn.ReLU(),
            torch.nn.Linear(16*channels,100)
        )
        
    def forward(self,x):
        return self.dense( self.conv(x).squeeze(-1).squeeze(-1))



invgan=DCGAN32Inverter(32)
invgan(batch[0]).shape

batch[1].shape

# ### Setting up training

# +
import torch
import torchvision
from PIL import Image
ngpu = 2
from matplotlib import pyplot as plt
import numpy as np
import torchvision.utils as vutils
from tqdm.autonotebook import tqdm
import os
import src.data.make_dataset as md
from src.models.util import GaussianNoise,MemBatch

device = torch.device("cuda:0")

device = torch.device("cuda")
os.chdir('/home/ndeutsch/face_gans/face_gans')
dataloader = datasets.create_dcgan32_inversion_dataloader(batch_size=512,num_workers=6)

invgan=DCGAN32Inverter(32).to(device)
invgan.to(device)
loss = torch.nn.MSELoss()
optim = torch.optim.Adam(invgan.parameters())

n_epochs = 1

Losses = []

for epoch in range(n_epochs):
    print("Epoch ",epoch)
    for i,data in enumerate(dataloader):
        imgs = data[0].to(device)
        vecs_tgt = data[1].to(device)

        optim.zero_grad()
        vecs = invgan(imgs)
        L = loss(vecs, vecs_tgt)
        L.backward()
        optim.step()
        Loss = L.detach().cpu().item()
        Losses.append(Loss)
        print("Epoch: ",epoch,"Step: ",i,"Loss: ",Loss)

# +
invgan=DCGAN32Inverter(32).to(device)
invgan.to(device)
loss = torch.nn.MSELoss()
optim = torch.optim.Adam(invgan.parameters())

n_epochs = 1

Losses = []

for epoch in range(n_epochs):
    print("Epoch ",epoch)
    for i,data in enumerate(dataloader):
        imgs = data[0].to(device)
        vecs_tgt = data[1].to(device)

        optim.zero_grad()
        vecs = invgan(imgs)
        L = loss(vecs, vecs_tgt)
        L.backward()
        optim.step()
        Loss = L.detach().cpu().item()
        Losses.append(Loss)
        print("Epoch: ",epoch,"Step: ",i,"Loss: ",Loss)
            
# -
# This is much too slow


