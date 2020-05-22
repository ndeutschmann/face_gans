"""Training functions for dcgan32"""
import os
import datetime as dt
import numpy as np
import torch
import torchvision.utils as vutils
import matplotlib.pyplot as plt
from src.models.dcgan32.dcgan32 import DCGen32, DCDiscr32
from src.models.util import GaussianNoise, MemBatch
from src.data.datasets import create_celebA_dataloader


def Dstep(G, D, optimD, DLoss, noise, data,
          mem, mem_update_size, mem_update_prob, mem_batch_size, device):

    for p in G.parameters():
        p.requires_grad = False
    for p in D.parameters():
        p.requires_grad = True

    # The dataloader yields a pair image,label. The label is useless
    true_batch, _ = data
    true_batch = noise(true_batch.to(device))

    batch_size = true_batch.size()[0]

    optimD.zero_grad()
    # First pass our true batch through the discriminator and estimate the loss
    # "true image label" = 0
    tgt_true_labels = torch.zeros(batch_size, device=device)
    true_labels = D(true_batch).squeeze()
    true_loss = DLoss(true_labels, tgt_true_labels)
    true_loss.backward()

    # Then we generate random noise vectors and obtain original (new) fake images
    x = torch.zeros(batch_size, 100, device=device).normal_(0, 1)
    new_fake_batch = noise(G(x))

    # We occasionally store some of the generated images in memory
    if np.random.rand() > mem_update_prob:
        mem.add(new_fake_batch[:mem_update_size])

    # We sample some images from previous iterations and add them to our fake batch
    mem_batch = mem.sample(mem_batch_size).to(device)
    fake_batch = torch.cat([new_fake_batch, mem_batch], 0)

    # The label for fake images is 1
    tgt_fake_labels = torch.ones(batch_size + mem_batch_size, device=device)
    fake_labels = D(fake_batch).squeeze()
    fake_loss = DLoss(fake_labels, tgt_fake_labels)
    fake_loss.backward()
    optimD.step()
    return true_loss.item(), fake_loss.item()


def Gstep(G, D, optimG, GLoss, noise, batch_size, device):
    x = torch.zeros(batch_size, 100, device=device).normal_(0, 1)
    for p in G.parameters():
        p.requires_grad = True
    for p in D.parameters():
        p.requires_grad = False
    optimG.zero_grad()
    batch = noise(G(x))
    target_labels = torch.zeros(batch_size, device=device)
    labels = D(batch).squeeze()

    loss = GLoss(labels, target_labels)
    loss.backward()
    optimG.step()
    return loss.item()


def train_dcgan32(n_epochs=100,
                  exp_root="tmp/dcgan32_training",
                  exp_id="1",
                  loss_report_period=150,
                  batch_size=128,
                  mem_batch_size=16,
                  mem_capacity=128,
                  mem_update_size=16,
                  mem_update_prob=.7,
                  learning_rate_g=3.e-5,
                  learning_rate_d=6.e-5,
                  noise_level=0.2,
                  noise_damping=1.5,
                  noise_damping_period=5,
                  data_root="celebA/unlabelled_centered",
                  num_workers=1,
                  device=None,
                  torch_seed=None,
                  checkpoint_period=10):

    assert torch.cuda.is_available(), "Must run on GPU"
    # Decide which device we want to run on
    if device is None:
        device = torch.device("cuda:0")

    os.makedirs(exp_root, exist_ok=True)
    exp_dir = os.path.join(exp_root,exp_id)
    try:
        os.makedirs(exp_dir,exist_ok=False)
    except FileExistsError as e:
        print("Experiments must run in a unique folder given by exp_root/exp_id")
        raise


    image_size = 32
    dataloader = create_celebA_dataloader(image_size,
                                          batch_size=batch_size,
                                          num_workers=num_workers,
                                          data_root=data_root)

    flosses = []
    tlosses = []
    glosses = []

    D = DCDiscr32().to(device)
    G = DCGen32().to(device)

    DLoss = torch.nn.BCELoss()
    GLoss = torch.nn.BCELoss()

    optimG = torch.optim.Adam(G.parameters(), lr=learning_rate_g, betas=(0.5, 0.999))
    optimD = torch.optim.Adam(D.parameters(), lr=learning_rate_d, betas=(0.5, 0.999))

    noise = GaussianNoise(noise_level)

    mem = MemBatch(mem_capacity, (3, image_size, image_size), device=torch.device("cpu"), noise_capacity=0)

    if torch_seed is not None:
        torch.manual_seed(torch_seed)

    for epoch in range(n_epochs):
        if epoch % noise_damping_period == 0 and epoch > 0:
            noise_level /= noise_damping
            noise = GaussianNoise(noise_level)

        for i, data in enumerate(dataloader):
            tloss, floss = Dstep(G, D, optimD, DLoss, noise, data,
                                 mem, mem_update_size, mem_update_prob, mem_batch_size, device)
            gloss = Gstep(G, D, optimG, GLoss, noise, batch_size, device)
            gloss = Gstep(G, D, optimG, GLoss, noise, batch_size, device)
            tlosses.append(tloss)
            flosses.append(floss)
            glosses.append(gloss)

            if (i % loss_report_period) == 0:
                print(f"step {i}| true loss = {tloss:.3e}| fake loss = {floss:.3e}| gen loss = {gloss:.3e}")

        ts = dt.datetime.timestamp(dt.datetime.now())
        torch.save({
            "timestamp": ts,
            "epoch": epoch,
            "tlosses": tlosses,
            "flosses": flosses,
            "glosses": glosses,
        }, os.path.join(exp_dir, "run_summary.pkl"))

        if epoch % checkpoint_period == 0:
            torch.save({
                'model_state_dict': G.state_dict(),
                'optimizer_state_dict': optimG.state_dict()
            }, os.path.join(exp_dir, f"checkpoint_G_epoch{epoch}.pkl"))

            torch.save({
                'model_state_dict': D.state_dict(),
                'optimizer_state_dict': optimD.state_dict()
            }, os.path.join(exp_dir, f"checkpoint_D_epoch{epoch}.pkl"))

            x = torch.zeros(64, 100, device=device).normal_(0, 1)
            batch = G(x)

            plt.figure(figsize=(6, 6))
            plt.imshow(np.transpose(vutils.make_grid(batch, padding=2, normalize=True).detach().cpu(), (1, 2, 0)))
            plt.savefig(os.path.join(exp_dir, f"samples_epoch{epoch}.png"))
