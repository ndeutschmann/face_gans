import torch
import src.data.datasets as datasets
import os
import time



os.chdir('/home/ndeutsch/face_gans/face_gans')
device = torch.device("cuda")


class DCGAN32Inverter(torch.nn.Module):
    def __init__(self, channels=32):
        super(DCGAN32Inverter, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.BatchNorm2d(3),
            torch.nn.Conv2d(3, channels, 3, 1, 1),
            torch.nn.BatchNorm2d(channels),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels, channels * 2, 3, 2, 1),
            torch.nn.BatchNorm2d(channels * 2),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels * 2, channels * 4, 2, 2, 1),
            torch.nn.BatchNorm2d(channels * 4),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels * 4, channels * 8, 2, 2),
            torch.nn.BatchNorm2d(channels * 8),
            torch.nn.ReLU(),
            torch.nn.Conv2d(channels * 8, channels * 16, 2, 4),
            torch.nn.BatchNorm2d(channels * 16),
            torch.nn.ReLU()
        )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(16 * channels, 16 * channels),
            torch.nn.BatchNorm1d(channels * 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16 * channels, 100)
        )

    def forward(self, x):
        return self.dense(self.conv(x).squeeze(-1).squeeze(-1))

batch_size = 256
dataloader = datasets.create_dcgan32_inversion_dataloader(batch_size=batch_size,num_workers=16,pin_memory=True)

invgan=DCGAN32Inverter(32).to(device)
invgan.to(device)
loss = torch.nn.MSELoss()
optim = torch.optim.Adam(invgan.parameters())

n_epochs = 1

Losses = []
t0 = time.time()
for epoch in range(n_epochs):
    print("Epoch ",epoch)
    for i,data in enumerate(dataloader):
        print("{} Im/s".format(batch_size / (time.time() - t0)))
        t0 = time.time()
        imgs = data[0].to(device)
        vecs_tgt = data[1].to(device)

        optim.zero_grad()
        vecs = invgan(imgs)
        L = loss(vecs, vecs_tgt)
        L.backward()
        optim.step()
        Loss = L.detach().cpu().item()
        Losses.append(Loss)
        print("Epoch: ",epoch,"Image: {:0.1f}k/100k".format(i*batch_size/1000.),"Loss: ",Loss)
