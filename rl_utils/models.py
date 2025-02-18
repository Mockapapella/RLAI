import torch.nn as nn
import torch
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint_sequential


class RocketNet(nn.Module):
    def __init__(self):
        super(RocketNet, self).__init__()
        # Input dimension reduction layer
        self.input_reduction = nn.Sequential(
            nn.Flatten(), nn.Linear(480 * 270 * 3, 1024), nn.ReLU(), nn.Dropout(0.5)
        )

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Binary controls (buttons) - no activation
        self.binary_head = nn.Linear(128, 11)

        # Analog controls - with sigmoid activation built-in
        self.analog_head = nn.Sequential(nn.Linear(128, 8), nn.Sigmoid())

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, a=0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0.01)

    def forward(self, x):
        # No autocast in forward pass
        x = self.input_reduction(x)
        features = self.backbone(x)

        # Get outputs from both heads
        binary_out = self.binary_head(features)
        analog_out = self.analog_head(features)

        # Concatenate results
        return torch.cat([binary_out, analog_out], dim=1)
