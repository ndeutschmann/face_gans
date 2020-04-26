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

nbatch = 128

x = torch.zeros(nbatch,100,device=device).normal_(0,1.)
batch = G(x)

batch


