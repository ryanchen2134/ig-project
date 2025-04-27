import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=128, backbone_type='resnet18'):
        """
        Initialize the image encoder with a choice of backbone.
        
        Args:
            embedding_dim: Dimension of the final embedding
            backbone_type: Type of backbone to use ('resnet18', 'efficientnet_b0', or 'convnext_tiny')
        """
        super(ImageEncoder, self).__init__()
        
        # Feature dimensions for different backbones
        feature_dims = {
            'resnet18': 512,
            'efficientnet_b0': 1280,
            'convnext_tiny': 768
        }
        
        # Load the selected backbone
        if backbone_type == 'resnet18':
            # ResNet18 - lightweight and effective for smaller datasets
            backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
            self.backbone = nn.Sequential(*list(backbone.children())[:-2])  # Remove avg pool and fc
            self.feature_dim = feature_dims['resnet18']
            
        elif backbone_type == 'efficientnet_b0':
            # EfficientNetB0 - efficient architecture with good performance
            backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.backbone = backbone.features  # Use the features part
            self.feature_dim = feature_dims['efficientnet_b0']
            
        elif backbone_type == 'convnext_tiny':
            # Original ConvNeXt implementation (kept for reference)
            backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self.backbone = backbone.features
            self.feature_dim = feature_dims['convnext_tiny']
            
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        # Projection Layer with Bottleneck - helps avoid overfitting
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 256),
            nn.GroupNorm(num_groups=8, num_channels=256),  # GroupNorm instead of BatchNorm
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, embedding_dim)
        )

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.projection(features)
        # Normalize embeddings to lie on unit hypersphere
        return F.normalize(embedding, p=2, dim=1)


if __name__ == "__main__":
    # Test all backbone options
    for backbone_type in ['resnet18', 'efficientnet_b0', 'convnext_tiny']:
        print(f"\nTesting {backbone_type}...")
        encoder = ImageEncoder(embedding_dim=64, backbone_type=backbone_type)
        encoder.eval()
        # Create a dummy input tensor with shape (batch_size, channels, height, width)
        dummy_input = torch.randn(1, 3, 224, 224)
        # Pass the dummy input through the encoder
        output = encoder(dummy_input)
        # Print the output shape and tensor
        print(f"Output shape: {output.shape}")
        print(f"Output norm: {torch.norm(output, dim=1)}")  # Should be close to 1.0