"""A regression model to invert dcgan32 (image -> latent space vector
It is hardcoded but works
"""
import torch


class DCGAN32Inverter(torch.nn.Module):
    def __init__(self, channels=32,dropout_rate=.3):
        super(DCGAN32Inverter, self).__init__()

        self.conv = torch.nn.Sequential(
            torch.nn.Conv2d(3, channels, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(channels),
            torch.nn.Conv2d(channels, channels * 2, 3, 2, 1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(channels * 2),
            torch.nn.Conv2d(channels * 2, channels * 4, 2, 2, 1),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(channels * 4),
            torch.nn.Dropout(dropout_rate),
            torch.nn.Conv2d(channels * 4, channels * 8, 2, 2),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(channels * 8),
            torch.nn.Conv2d(channels * 8, channels * 16, 2, 4),
            torch.nn.LeakyReLU(),
            torch.nn.BatchNorm2d(channels * 16),
        )

        self.dense = torch.nn.Sequential(
            torch.nn.Linear(16 * channels, 400),
            torch.nn.LeakyReLU(),
            torch.nn.Dropout(dropout_rate), 
            torch.nn.BatchNorm1d(400),
            torch.nn.Linear(400, 100)
        )

    def forward(self, x):
        return self.dense(self.conv(x).squeeze(-1).squeeze(-1))


class DCgan32ResnetInverter(torch.nn.Module):
    def __init__(self, n_channels=64):
        super(DCgan32ResnetInverter, self).__init__()

        # Input size 32
        self.pre_block1 = torch.nn.Conv2d(3, n_channels, 3, 1, 1)
        self.block1 = torch.nn.Sequential(
            torch.nn.Conv2d(n_channels, n_channels, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(n_channels, n_channels, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(n_channels, n_channels, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(n_channels, n_channels, 3, 1, 1),
            torch.nn.LeakyReLU(),
        )

        # Input size 16
        self.pre_block2 = torch.nn.Conv2d(n_channels, 2 * n_channels, 1, 1, 0)
        self.block2 = torch.nn.Sequential(
            torch.nn.Conv2d(2 * n_channels, 2 * n_channels, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(2 * n_channels, 2 * n_channels, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(2 * n_channels, 2 * n_channels, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(2 * n_channels, 2 * n_channels, 3, 1, 1),
            torch.nn.LeakyReLU(),
        )

        # Input size 8
        self.pre_block3 = torch.nn.Conv2d(2 * n_channels, 4 * n_channels, 1, 1, 0)
        self.block3 = torch.nn.Sequential(
            torch.nn.Conv2d(4 * n_channels, 4 * n_channels, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(4 * n_channels, 4 * n_channels, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(4 * n_channels, 4 * n_channels, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(4 * n_channels, 4 * n_channels, 3, 1, 1),
            torch.nn.LeakyReLU(),
        )

        # Input size 4
        self.pre_block4 = torch.nn.Conv2d(4 * n_channels, 4 * n_channels, 1, 1, 0)
        self.block4 = torch.nn.Sequential(
            torch.nn.Conv2d(4 * n_channels, 4 * n_channels, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(4 * n_channels, 4 * n_channels, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(4 * n_channels, 4 * n_channels, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(4 * n_channels, 4 * n_channels, 3, 1, 1),
            torch.nn.LeakyReLU(),
            torch.nn.Conv2d(4 * n_channels, 4 * n_channels, 1, 1, 0),
            torch.nn.LeakyReLU(),
        )

        self.max_pool = torch.nn.MaxPool2d(2)
        self.avg_pool = torch.nn.AvgPool2d(4)

        self.classifier = torch.nn.Linear(4 * n_channels, 100)

    def forward(self, x):
        x = self.pre_block1(x)
        x = self.block1(x) + x
        x = self.max_pool(x)
        x = self.pre_block2(x)
        x = self.block2(x) + x
        x = self.max_pool(x)
        x = self.pre_block3(x)
        x = self.block3(x) + x
        x = self.max_pool(x)
        x = self.pre_block4(x)
        x = self.block4(x) + x
        x = self.avg_pool(x).squeeze(-1).squeeze(-1)
        x = self.classifier(x)

        return x
