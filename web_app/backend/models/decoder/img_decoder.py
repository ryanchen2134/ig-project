import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=128, backbone_type='resnet18', freeze_layers=4):
        """
        Initialize the image encoder with a choice of backbone.
        
        Args:
            embedding_dim: Dimension of the final embedding
            backbone_type: Type of backbone to use ('resnet18', 'efficientnet_b0', or 'convnext_tiny')
            freeze_layers: Number of early layers to freeze (ResNet has 8 main layers)
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
            
            # Get the individual layers 
            backbone_layers = list(backbone.children())
            self.backbone = nn.Sequential(*backbone_layers[:-2])  # Remove avg pool and fc
            self.feature_dim = feature_dims['resnet18']
            
            # Only freeze early layers (typically conv1, bn1, and some residual blocks)
            # ResNet main blocks: [conv1, bn1, relu, maxpool, layer1, layer2, layer3, layer4]
            if freeze_layers > 0:
                for i, child in enumerate(self.backbone.children()):
                    if i < freeze_layers:  # Freeze only the specified number of layers
                        for param in child.parameters():
                            param.requires_grad = False
                    else:
                        # Enable gradient for later layers
                        for param in child.parameters():
                            param.requires_grad = True
            
        elif backbone_type == 'efficientnet_b0':
            # EfficientNetB0 - efficient architecture with good performance
            backbone = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
            self.backbone = backbone.features  # Use the features part
            self.feature_dim = feature_dims['efficientnet_b0']
            
            # Freeze early layers (freeze only first few blocks)
            if freeze_layers > 0:
                # EfficientNet has 8 blocks
                total_blocks = len(self.backbone)
                blocks_to_freeze = min(freeze_layers, total_blocks - 2)  # Leave at least 2 blocks unfrozen
                
                for i, block in enumerate(self.backbone):
                    if i < blocks_to_freeze:
                        for param in block.parameters():
                            param.requires_grad = False
                    else:
                        for param in block.parameters():
                            param.requires_grad = True
            
        elif backbone_type == 'convnext_tiny':
            # ConvNeXt implementation
            backbone = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
            self.backbone = backbone.features
            self.feature_dim = feature_dims['convnext_tiny']
            
            # Freeze early stages (ConvNeXt has multiple stages)
            if freeze_layers > 0:
                # ConvNeXt has 4 stages - freeze a proportion based on freeze_layers
                stages_to_freeze = min(freeze_layers // 2, 3)  # Leave at least one stage unfrozen
                
                for i, stage in enumerate(self.backbone):
                    if i < stages_to_freeze:
                        for param in stage.parameters():
                            param.requires_grad = False
                    else:
                        for param in stage.parameters():
                            param.requires_grad = True
            
        else:
            raise ValueError(f"Unsupported backbone type: {backbone_type}")
        
        # LOCKED 
        self.pool = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten()
        )
        
        # UNLOCKED projection layers
        self.projection1 = nn.Linear(self.feature_dim, embedding_dim)  # First UNLOCKED layer
        
        # Nonlinear transformation
        self.nonlinear = nn.Sequential(
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        # Second UNLOCKED layer
        self.projection2 = nn.Linear(embedding_dim, embedding_dim)
        
    def forward(self, x):
        # Process through backbone (some layers may be frozen, others trainable)
        features = self.backbone(x)
        pooled = self.pool(features)
        
        # Unlocked projections
        embedding = self.projection1(pooled)
        embedding = self.nonlinear(embedding)
        embedding = self.projection2(embedding)
        
        # Normalize
        return F.normalize(embedding, p=2, dim=1)