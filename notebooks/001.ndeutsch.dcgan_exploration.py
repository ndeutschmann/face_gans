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

device = torch.device("cuda")

example = Image.open("../data/processed/img_align_celeba/000002.jpg")

example 

dataroot = "../data/processed"
image_size = 32
batch_size = 128
workers = 2

dataloader = md.create_celebA_dataloader(32,data_root=dataroot)

# +
# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))


# -

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

cg = ConvG()
x = torch.zeros(12,100).normal_(0,1)
out = cg(x)
print(out.shape)
plt.imshow(np.transpose(torch.sigmoid(torch.squeeze(out[0]).detach()).numpy(),(1,2,0)))


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
#            torch.nn.Linear(32,32),
#            torch.nn.BatchNorm1d(32),
#            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(256,1),
            torch.nn.Sigmoid()            
        )
        
        
    def forward(self,x):
        z = self.conv(x)
        return self.dense(z.view(-1,256))

# +
myMem=MemBatch(32,(3,image_size,image_size),device=torch.device("cpu"),noise_capacity=4)
cg = ConvG()
x = torch.zeros(16,100).normal_(0,1)
out = cg(x)
myMem.add(out)
out = (next(iter(dataloader)))[0][:32]
myMem.add(out)
x = torch.zeros(16,100).normal_(0,1)
out = cg(x)
myMem.add(out)

Batch=myMem.sample(32)

plt.imshow(np.transpose(vutils.make_grid(Batch, padding=2, normalize=True).detach().cpu(),(1,2,0)))
out[0].shape
# -

myD=ConvD().cuda()
myG=ConvG().cuda()
noise=GaussianNoise(.05).cuda()

optimG = torch.optim.Adam(myG.parameters(),lr=2.e-4,betas=(0.5, 0.999))
optimD = torch.optim.Adam(myD.parameters(),lr=2.e-4,betas=(0.5, 0.999))
DLoss = torch.nn.BCELoss()
GLoss = torch.nn.BCELoss()


def Dstep(G,D,optimD,DLoss,noise,data,mem,mem_batch):
    for p in G.parameters():
        p.requires_grad=False
    for p in D.parameters():
        p.requires_grad=True        
    
    optimD.zero_grad()
    
    true_batch,_ = data
    true_batch = noise(true_batch.to(device))
    tgt_true_labels = torch.zeros(true_batch.size()[0],device=device)
    true_labels = D(true_batch).squeeze()
    true_loss = DLoss(true_labels,tgt_true_labels)
    true_loss.backward()
    
    x = torch.zeros(batch_size,100,device=device).normal_(0,1)
    new_fake_batch = noise(G(x))
    if np.random.rand() > .7:
        mem.add(new_fake_batch[:mem_batch])
    
    mem_batch = mem.sample(batch_size//8).to(device)
    fake_batch = torch.cat([new_fake_batch,mem_batch],0)
    tgt_fake_labels = torch.ones(batch_size+batch_size//8,device=device)
    fake_labels = D(fake_batch).squeeze()
    fake_loss = DLoss(fake_labels,tgt_fake_labels)
    fake_loss.backward()
    optimD.step()
    return true_loss.item(),fake_loss.item()


def Gstep(G,D,optimG,GLoss,noise):
    x = torch.zeros(batch_size,100,device=device).normal_(0,1)
    for p in G.parameters():
        p.requires_grad=True
    for p in D.parameters():
        p.requires_grad=False
    optimG.zero_grad()        
    Batch = noise(G(x))
    TargetLabels = torch.zeros(batch_size,device=device)
    Labels = D(Batch).squeeze()
    
    loss = GLoss(Labels,TargetLabels)
    loss.backward()
    optimG.step()
    return loss.item()


flosses=[]
tlosses = []
glosses=[]
myD=ConvD().cuda()
myG=ConvG().cuda()
noise=GaussianNoise(0.05)
optimG = torch.optim.Adam(myG.parameters(),lr=1.e-4,betas=(0.5, 0.999))
optimD = torch.optim.Adam(myD.parameters(),lr=1.e-4,betas=(0.5, 0.999))
data_iter = iter(dataloader)

# !rm -rf exp24_dense

# +
expdir = "exp24_dense"
os.makedirs(expdir)
for attempt in range(1):
    lrG = 3.e-5
    lrD = 6.e-5
    flosses=[]
    tlosses = []
    glosses=[]
    myD=ConvD().cuda()
    myG=ConvG().cuda()
    noiselevel = 0.2
    noise=GaussianNoise(noiselevel)
    myMem = MemBatch(batch_size,(3,image_size,image_size),device=torch.device("cpu"),noise_capacity=0)
    mem_batch = 16
    optimG = torch.optim.Adam(myG.parameters(),lr=lrG,betas=(0.5, 0.999))
    optimD = torch.optim.Adam(myD.parameters(),lr=lrD,betas=(0.5, 0.999))
    for epoch in range(100):
        if epoch % 5 == 0:
            noiselevel/=1.5
            noise=GaussianNoise(noiselevel)
        for i,data in tqdm(enumerate(dataloader)):
            tloss,floss = (Dstep(myG,myD,optimD,DLoss,noise,data,myMem,mem_batch))    
            gloss = Gstep(myG,myD,optimG,GLoss,noise)  
            gloss = Gstep(myG,myD,optimG,GLoss,noise)              
            tlosses.append(tloss)
            flosses.append(floss)
            glosses.append(gloss)            
        

        x = torch.zeros(64,100,device=device).normal_(0,1)
        Batch= myG(x)

        plt.figure(figsize=(6,6))
        plt.imshow(np.transpose(vutils.make_grid(Batch, padding=2, normalize=True).detach().cpu(),(1,2,0)))
        plt.savefig("{expdir}/attempt{attempt}.epoch{epoch}.{lrG}.{lrD}.png".format(expdir=expdir,attempt=attempt,epoch=epoch,lrG=lrG,lrD=lrD))
        plt.show()
        fig = plt.figure()
        plt.plot(tlosses,label="D - t")
        plt.plot(flosses,label="D - f")
        plt.plot(glosses,label="G")
        fig.legend()
        plt.show()  
    
    fig = plt.figure()
    plt.plot(tlosses,label="D - t")
    plt.plot(flosses,label="D - f")
    plt.plot(glosses,label="G")
    fig.legend()
    plt.savefig("{expdir}/curves.attempt{attempt}.epoch{epoch}.{lrG}.{lrD}.png".format(expdir=expdir,attempt=attempt,epoch=epoch,lrG=lrG,lrD=lrD))
    plt.show()
    torch.save({
            'model_state_dict': myG.state_dict(),
            'optimizer_state_dict': optimG.state_dict(),
            }, "{expdir}/checkpointG.attempt{attempt}.epoch{epoch}.{lrG}.{lrD}".format(expdir=expdir,attempt=attempt,epoch=epoch,lrG=lrG,lrD=lrD))
    torch.save({
            'model_state_dict': myD.state_dict(),
            'optimizer_state_dict': optimD.state_dict(),
            }, "{expdir}/checkpointD.attempt{attempt}.epoch{epoch}.{lrG}.{lrD}".format(expdir=expdir,attempt=attempt,epoch=epoch,lrG=lrG,lrD=lrD))
    



# +
x = torch.zeros(64,100,device=device).normal_(0,1)
Batch= myG(x)

plt.figure(figsize=(5,5))
plt.imshow(np.transpose(vutils.make_grid(Batch, padding=2, normalize=True).detach().cpu(),(1,2,0)))
plt.show()

# -




