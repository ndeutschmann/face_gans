import torch


class DCGen32(torch.nn.Module):
    """Generator architecture for 100->32x32 images"""
    def __init__(self):
        super(DCGen32, self).__init__()
        self.deconv = torch.nn.Sequential(
            torch.nn.ConvTranspose2d(100, 256, 4, 1),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(256, 256, 3, 1, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(256, 128, 4, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(128, 64, 4, 2),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(64, 64, 3, 2, 1),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(64, 32, 3, 1, 1),
            torch.nn.BatchNorm2d(32),
            torch.nn.LeakyReLU(0.2),
            torch.nn.ConvTranspose2d(32, 3, 2, 1),
            torch.nn.Tanh()
        )
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(100, 100),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Linear(100, 100),
            torch.nn.BatchNorm1d(100)
        )

    def forward(self, x):
        z = (self.dense(x) + x).unsqueeze(-1).unsqueeze(-1)
        return self.deconv(z)


class DCDiscr32(torch.nn.Module):
    """Discriminator architecture for 32x32 images"""
    def __init__(self):
        super(DCDiscr32, self).__init__()
        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, 2, 2),
            torch.nn.BatchNorm2d(16),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(16, 64, 4, 2),
            torch.nn.BatchNorm2d(64),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(64, 128, 4, 1),
            torch.nn.BatchNorm2d(128),
            torch.nn.LeakyReLU(0.2),
            torch.nn.Conv2d(128, 256, 4, 1),
            torch.nn.BatchNorm2d(256),
            torch.nn.LeakyReLU(0.2),
        )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(256, 1),
            torch.nn.Sigmoid()
        )

    def forward(self, x):
        z = self.conv(x)
        return self.dense(z.view(-1, 256))