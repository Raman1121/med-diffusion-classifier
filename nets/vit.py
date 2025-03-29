import torch
import torch.nn as nn
from typing import Optional
import timm

class ViT2D(nn.Module):
    """
    A ViT-based backbone for 2D image inputs, returning feature embeddings
    that can be used for downstream classification or other tasks.
    """

    def __init__(
        self,
        variant: str = "vit_small_patch16_224",
        pretrained: bool = True,
        in_channels: int = 3,
        img_size: Optional[int] = (224, 224),
    ):
        """
        Args:
            variant (str): Which ViT variant to load from timm 
                (e.g., 'vit_small_patch16_224', 'vit_medium_patch16_224', 'vit_base_patch16_224', etc.).
            pretrained (bool): Whether to load pretrained ImageNet weights.
            in_channels (int): Number of input channels (default=3 for RGB).
        """
        
        super().__init__()
        self.variant = variant
        self.in_channels = in_channels

        self.vit = timm.create_model(
            model_name=variant,
            pretrained=pretrained,
            in_chans=in_channels,
            img_size=img_size,
        )

        if hasattr(self.vit, "head") and hasattr(self.vit.head, "in_features"):
            self.output_dim = self.vit.head.in_features
        else:
            raise ValueError(f"Cannot determine output_dim for variant: {variant}")

        # Remove the classification head to get raw features
        self.vit.head = nn.Identity()

        # print(self.vit)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through ViT (minus the final classification head).
        
        Args:
            x (torch.Tensor): Input tensor of shape [B, in_channels, H, W].
        
        Returns:
            features (torch.Tensor): Feature embeddings of shape [B, output_dim].
        """
        features = self.vit(x)  # shape [B, output_dim]
        return features