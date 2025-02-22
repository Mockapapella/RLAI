"""Neural network model and loss functions for Rocket League AI."""

import torch
import torch.nn as nn
import torch.nn.functional as F


class RocketNet(nn.Module):
    """Neural network for predicting Rocket League controller inputs from game frames.

    Architecture:
        - Shared convolutional backbone for processing input frames
        - Shared fully connected backbone
        - Separate heads for binary (buttons) and analog (sticks/triggers) outputs
    """

    def __init__(self):
        super(RocketNet, self).__init__()

        # Use convolutional layers to efficiently process the input
        self.conv_backbone = nn.Sequential(
            # First conv block - reduce spatial dimensions
            nn.Conv2d(3, 32, kernel_size=7, stride=4, padding=3),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(2),
            # Second conv block - extract features
            nn.Conv2d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(2),
            # Third conv block - final feature extraction
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.AdaptiveAvgPool2d((4, 4)),  # Force output to 4x4 spatial dimensions
        )

        # Calculate the flattened size after convolutions (128 * 4 * 4 = 2048)
        self._conv_output_size = 128 * 4 * 4

        # Shared backbone
        self.backbone = nn.Sequential(
            nn.Linear(self._conv_output_size, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.BatchNorm1d(256),
            nn.Dropout(0.2),
            nn.Linear(256, 128),
            nn.ReLU(),
        )

        # Binary controls (buttons) - no activation
        self.binary_head = nn.Linear(128, 11)

        # Improved analog head with more expressive architecture
        self.analog_head = nn.Sequential(
            nn.Linear(128, 64),
            nn.LayerNorm(64),  # Normalize activations
            nn.LeakyReLU(0.1),
            nn.Dropout(0.1),  # Prevent overfitting
            nn.Linear(64, 8),
            nn.Sigmoid(),
        )

        # Apply weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Linear, nn.Conv2d)):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Process input frames through the network.

        Args:
            x: Input tensor of shape (B, C, H, W) or (B, H, W, C).

        Returns:
            Tensor of shape (B, 19) containing predicted controller inputs.
        """
        # Ensure proper input format (B, C, H, W) for convolutions
        if x.dim() == 4 and x.shape[1] == 270 and x.shape[2] == 480:
            # Input is (B, H, W, C), need to permute to (B, C, H, W)
            x = x.permute(0, 3, 1, 2)

        # Process through convolutional layers
        x = self.conv_backbone(x)

        # Flatten - use reshape instead of view to handle non-contiguous tensors
        x = x.reshape(x.size(0), -1)

        # Process through shared backbone
        features = self.backbone(x)

        # Get outputs from both heads
        binary_out = self.binary_head(features)
        analog_out = self.analog_head(features)

        # Concatenate results
        return torch.cat([binary_out, analog_out], dim=1)


class FocalLoss(nn.Module):
    """Focal Loss to address class imbalance for binary controls."""

    def __init__(self, gamma=2.0, alpha=0.25):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss between predictions and targets.

        Args:
            logits: Raw model outputs before sigmoid.
            targets: Binary target values.

        Returns:
            Scalar loss value.
        """
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)

        # Get the binary cross entropy loss
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")

        # Calculate focal weights
        p_t = probs * targets + (1 - probs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma

        # Apply weights to BCE loss (removed 1.5x multiplier)
        loss = focal_weight * bce_loss

        return loss.mean()


def compute_improved_loss(outputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Compute combined loss for binary and analog controls.

    Uses focal loss for binary controls (buttons) and smooth L1 loss for analog
    controls (sticks/triggers), with additional regularization for analog outputs.

    Args:
        outputs: Model predictions of shape (B, 19).
        targets: Target values of shape (B, 19).

    Returns:
        Combined loss value as a scalar tensor.
    """
    # Split the outputs and targets
    binary_logits = outputs[:, :11]  # First 11 are binary controls
    analog_values = outputs[:, 11:]  # Last 8 are analog controls

    binary_targets = targets[:, :11]
    analog_targets = targets[:, 11:]

    # Add epsilon for numerical stability
    eps = 1e-7

    # Use focal loss for binary controls
    focal_loss = FocalLoss(gamma=2.0, alpha=0.25)
    binary_loss = focal_loss(binary_logits, binary_targets)

    # Ensure analog values are in valid range
    analog_values = torch.clamp(analog_values, eps, 1.0 - eps)

    # Use smooth L1 loss for analog controls
    analog_loss = F.smooth_l1_loss(analog_values, analog_targets, reduction="mean", beta=0.1)

    # Add distribution regularization for analog outputs
    analog_distribution_loss = 0.0
    if analog_values.size(0) > 1:  # Only if batch size > 1
        # Encourage diversity in predictions across the batch
        analog_std = torch.clamp(analog_values.std(dim=0, unbiased=False), min=0.1)
        analog_distribution_loss = 0.05 * (1.0 / analog_std).mean()

    # Total loss with increased analog weight and distribution regularization
    total_loss = binary_loss + 3.0 * analog_loss + analog_distribution_loss

    # Final safety check
    if torch.isnan(total_loss) or torch.isinf(total_loss):
        return torch.tensor(0.1, device=total_loss.device, requires_grad=True)

    return total_loss
