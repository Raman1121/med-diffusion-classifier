import torch
import torch.nn as nn
from typing import Optional
import timm

class Swin2D(nn.Module):
    """
    A Swin Transformer-based backbone for 2D image inputs, returning feature embeddings
    that can be used for downstream classification or other tasks.
    """

    def __init__(
        self,
        variant: str = "swin_base_patch4_window7_224",
        pretrained: bool = True,
        in_channels: int = 3,
        img_size: Optional[int] = (224, 224),
    ):
        """
        Args:
            variant (str): Which Swin Transformer variant to load from timm 
                (e.g., 'swin_tiny_patch4_window7_224', 'swin_base_patch4_window7_224', etc.).
            pretrained (bool): Whether to load pretrained ImageNet weights.
            in_channels (int): Number of input channels (default=3 for RGB).
            img_size (Optional[int]): Input image size (default=(224, 224)).
        """
        
        super().__init__()
        self.variant = variant
        self.in_channels = in_channels

        self.swin = timm.create_model(
            model_name=variant,
            pretrained=pretrained,
            in_chans=in_channels,
            img_size=img_size,
        )

        if hasattr(self.swin, "head") and hasattr(self.swin.head, "in_features"):
            self.output_dim = self.swin.head.in_features
        else:
            raise ValueError(f"Cannot determine output_dim for variant: {variant}")

        # Remove the classification head to get raw features
        self.swin.head.fc = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through Swin Transformer (minus the final classification head).
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, in_channels, H, W].
        
        Returns:
            features (torch.Tensor): Feature embeddings of shape [B, output_dim].
        """
        features = self.swin(x)  # shape [B, output_dim]
        return features
