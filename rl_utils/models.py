import torch.nn as nn
from torch.amp import autocast
from torch.utils.checkpoint import checkpoint_sequential


class RocketNet(nn.Module):
    def __init__(self):
        super(RocketNet, self).__init__()
        # Input dimension reduction layer
        self.input_reduction = nn.Sequential(
            nn.Flatten(), nn.Linear(480 * 270 * 3, 1024), nn.ReLU(), nn.Dropout(0.5)
        )

        # Main network with slimmer layers
        self.main = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 19),
        )

        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights for better training stability"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                # Use a smaller initialization scale for better stability
                nn.init.kaiming_normal_(module.weight, a=0.1)
                if module.bias is not None:
                    # Initialize biases to small positive values for ReLU
                    nn.init.uniform_(module.bias, 0.01, 0.1)

    def forward(self, x):
        # Enable automatic mixed precision
        with autocast("cuda"):
            x = self.input_reduction(x)
            # Apply gradient checkpointing in forward pass with use_reentrant=False
            x = checkpoint_sequential(
                self.main, segments=3, input=x, use_reentrant=False
            )
            return x

    def freeze_input_reduction(self):
        """Freeze input reduction layers to save memory after initial training"""
        for param in self.input_reduction.parameters():
            param.requires_grad = False
