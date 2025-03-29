import torch
import torch.nn as nn
from timm import create_model

class EfficientNet2D(nn.Module):
    """
    A backbone based on EfficientNet for 2D image inputs, returning feature embeddings
    that can be used for downstream classification or tasks.
    """

    def __init__(
        self,
        variant: str = "efficientnet_b0",
        pretrained: bool = True,
        in_channels: int = 3,
    ):
        """
        Args:
            variant (str): Which EfficientNet variant to load (e.g., 'efficientnet_b0', 'efficientnet_b1', etc.).
            pretrained (bool): Whether to load pretrained weights.
            in_channels (int): Number of input channels (default=3 for RGB).
        """
        super().__init__()

        self.variant = variant
        self.in_channels = in_channels

        # Create EfficientNet model from timm
        self.efficientnet = create_model(self.variant, pretrained=pretrained, in_chans=in_channels, num_classes=0) # num_classes=0 to exclude final classification head
        self.output_dim = self.efficientnet.num_features  # Get the output feature dimension

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through EfficientNet (excluding final classification head).

        Args:
            x (torch.Tensor): Input image tensor of shape [B, in_channels, H, W].

        Returns:
            features (torch.Tensor): Feature embeddings of shape [B, output_dim].
        """
        # Pass through backbone
        features = self.efficientnet(x)

        return features
