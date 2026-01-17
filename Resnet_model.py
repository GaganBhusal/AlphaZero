import torch
import torch.nn as nn

class Resnet(nn.Module):

    def __init__(self, in_channels, out_channels, resnet_blocks = 5):
        super(Resnet, self).__init__()
        self.start = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

        self.blocks = nn.ModuleList(
            [ResidualConnection(out_channels) for _ in range(resnet_blocks)]
        )

        self.policy_head = nn.Sequential(
            nn.Conv2d(out_channels, out_channels = 32, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 6 * 7, 7)
        )

        self.value_head = nn.Sequential(
            nn.Conv2d(out_channels, out_channels = 3, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * 6 * 7, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Tanh()
        )

    def forward(self, x):
        x = self.start(x)
        for block in self.blocks:
            x = block(x)
        policy = self.policy_head(x)
        value = self.value_head(x)
        return policy, value
        

class ResidualConnection(nn.Module):

    def __init__(self, out_channels):
        super(ResidualConnection, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels = out_channels, out_channels = out_channels, kernel_size = 3, padding = 1),
            nn.BatchNorm2d(out_channels)     
        )

    def forward(self, x):
        features = self.block(x)
        return torch.relu(features + x)