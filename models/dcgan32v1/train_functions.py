import torch
import numpy as np


def Dstep(G, D, optimD, DLoss, noise, data, mem, batch_size, mem_batch,device):
    for p in G.parameters():
        p.requires_grad = False
    for p in D.parameters():
        p.requires_grad = True

    optimD.zero_grad()

    true_batch, _ = data
    true_batch = noise(true_batch.to(device))
    tgt_true_labels = torch.zeros(true_batch.size()[0], device=device)
    true_labels = D(true_batch).squeeze()
    true_loss = DLoss(true_labels, tgt_true_labels)
    true_loss.backward()

    x = torch.zeros(batch_size, 100, device=device).normal_(0, 1)
    new_fake_batch = noise(G(x))
    if np.random.rand() > .7:
        mem.add(new_fake_batch[:mem_batch])

    mem_batch = mem.sample(batch_size // 8).to(device)
    fake_batch = torch.cat([new_fake_batch, mem_batch], 0)
    tgt_fake_labels = torch.ones(batch_size + batch_size // 8, device=device)
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
    Batch = noise(G(x))
    TargetLabels = torch.zeros(batch_size, device=device)
    Labels = D(Batch).squeeze()

    loss = GLoss(Labels, TargetLabels)
    loss.backward()
    optimG.step()
    return loss.item()
