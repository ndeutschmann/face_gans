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