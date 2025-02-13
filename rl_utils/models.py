import torch.nn as nn
import torch.nn.functional as F


class RocketNet(nn.Module):
    def __init__(self):
        super(RocketNet, self).__init__()
        self.flatten = nn.Flatten(1, -1)
        self.rocket_net = nn.Sequential(
            nn.Linear(1280 * 720, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 16),
        )

    def forward(self, x):
        x = self.flatten(x)
        x = self.rocket_net(x)
        return x
