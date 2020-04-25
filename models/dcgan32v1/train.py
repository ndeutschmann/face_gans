from .model import ConvD,ConvG
from .train_functions import Gstep,Dstep
import torch
from src.models.util import GaussianNoise,MemBatch
from src.data.make_dataset import create_celebA_dataloader
from matplotlib import pyplot as plt
import os
from tqdm.autonotebook import tqdm
import numpy as np
import torchvision.utils as vutils

device = torch.device("cuda:0")

image_size = 32
batch_size = 128

flosses = []
tlosses = []
glosses=[]
myD=ConvD().cuda()
myG=ConvG().cuda()
noise=GaussianNoise(0.05)
optimG = torch.optim.Adam(myG.parameters(),lr=1.e-4,betas=(0.5, 0.999))
optimD = torch.optim.Adam(myD.parameters(),lr=1.e-4,betas=(0.5, 0.999))
DLoss = torch.nn.BCELoss()
GLoss = torch.nn.BCELoss()

dataloader = create_celebA_dataloader(image_size,data_root="../../data/processed")

expdir = "experiment_1"
os.makedirs(expdir)
for attempt in range(1):
    lrG = 3.e-5
    lrD = 6.e-5
    flosses = []
    tlosses = []
    glosses = []
    myD = ConvD().cuda()
    myG = ConvG().cuda()
    noiselevel = 0.2
    noise = GaussianNoise(noiselevel)
    myMem = MemBatch(batch_size, (3, image_size, image_size), device=torch.device("cpu"), noise_capacity=0)
    mem_batch = 16
    optimG = torch.optim.Adam(myG.parameters(), lr=lrG, betas=(0.5, 0.999))
    optimD = torch.optim.Adam(myD.parameters(), lr=lrD, betas=(0.5, 0.999))
    for epoch in range(100):
        if epoch % 5 == 0:
            noiselevel /= 1.5
            noise = GaussianNoise(noiselevel)
        for i, data in tqdm(enumerate(dataloader)):
            tloss, floss = (Dstep(myG, myD, optimD, DLoss, noise, data, myMem, mem_batch))
            gloss = Gstep(myG, myD, optimG, GLoss, noise)
            gloss = Gstep(myG, myD, optimG, GLoss, noise)
            tlosses.append(tloss)
            flosses.append(floss)
            glosses.append(gloss)

        x = torch.zeros(64, 100, device=device).normal_(0, 1)
        Batch = myG(x)

        plt.figure(figsize=(6, 6))
        plt.imshow(np.transpose(vutils.make_grid(Batch, padding=2, normalize=True).detach().cpu(), (1, 2, 0)))
        plt.savefig(
            "{expdir}/attempt{attempt}.epoch{epoch}.{lrG}.{lrD}.png".format(expdir=expdir, attempt=attempt, epoch=epoch,
                                                                            lrG=lrG, lrD=lrD))

    fig = plt.figure()
    plt.plot(tlosses, label="D - t")
    plt.plot(flosses, label="D - f")
    plt.plot(glosses, label="G")
    fig.legend()
    plt.savefig("{expdir}/curves.attempt{attempt}.epoch{epoch}.{lrG}.{lrD}.png".format(expdir=expdir, attempt=attempt,
                                                                                       epoch=epoch, lrG=lrG, lrD=lrD))
    torch.save({
        'model_state_dict': myG.state_dict(),
        'optimizer_state_dict': optimG.state_dict(),
    }, "{expdir}/checkpointG.attempt{attempt}.epoch{epoch}.{lrG}.{lrD}".format(expdir=expdir, attempt=attempt,
                                                                               epoch=epoch, lrG=lrG, lrD=lrD))
    torch.save({
        'model_state_dict': myD.state_dict(),
        'optimizer_state_dict': optimD.state_dict(),
    }, "{expdir}/checkpointD.attempt{attempt}.epoch{epoch}.{lrG}.{lrD}".format(expdir=expdir, attempt=attempt,
                                                                               epoch=epoch, lrG=lrG, lrD=lrD))

