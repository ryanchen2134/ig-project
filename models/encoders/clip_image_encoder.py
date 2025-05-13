import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from typing import Optional

class CLIPImageEncoder(nn.Module):
    """
    Image encoder using the CLIP ViT model with frozen weights
    """
    def __init__(self, 
                 clip_model_name: str = "ViT-B/32", 
                 embedding_dim: int = 128,
                 freeze_base: bool = True):
        """
        Initialize CLIP-based image encoder.
        
        Args:
            clip_model_name: CLIP model variant ('ViT-B/32', 'ViT-B/16', or 'ViT-L/14')
            embedding_dim: Dimension of the final embedding
            freeze_base: Whether to freeze the base CLIP model
        """
        super(CLIPImageEncoder, self).__init__()
        
        # Load CLIP model
        self.clip_model, self.preprocess = clip.load(clip_model_name, device="cpu")
        
        # Get feature dimension from CLIP model
        if "ViT-B/32" in clip_model_name:
            self.feature_dim = 512
        elif "ViT-B/16" in clip_model_name:
            self.feature_dim = 512
        elif "ViT-L/14" in clip_model_name:
            self.feature_dim = 768
        else:
            raise ValueError(f"Unsupported CLIP model: {clip_model_name}")
            
        # Freeze CLIP model weights if specified
        if freeze_base:
            for param in self.clip_model.parameters():
                param.requires_grad = False
                
        # Create projection layer to map CLIP features to embedding space
        self.projection = nn.Sequential(
            nn.Linear(self.feature_dim, embedding_dim),
            nn.LayerNorm(embedding_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(embedding_dim, embedding_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the CLIP image encoder.
        
        Args:
            x: Batch of images [B, C, H, W]
            
        Returns:
            Normalized embeddings [B, embedding_dim]
        """
        with torch.set_grad_enabled(self.clip_model.visual.transformer.resblocks[0].attn.in_proj_weight.requires_grad):
            # Extract features from CLIP's vision encoder
            features = self.clip_model.encode_image(x)
            
        # Apply projection to get embeddings
        embedding = self.projection(features)
        
        # Normalize embeddings
        return F.normalize(embedding, p=2, dim=1)
    
    def encode_from_path(self, image_path: str, device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Encode a single image from file path
        
        Args:
            image_path: Path to the image file
            device: Device to use for encoding
            
        Returns:
            Normalized embedding [1, embedding_dim]
        """
        from PIL import Image
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image_input = self.preprocess(image).unsqueeze(0)
        
        if device is not None:
            image_input = image_input.to(device)
            
        # Move to the same device as input
        if device is not None and next(self.parameters()).device != device:
            self.to(device)
            
        # Encode
        with torch.no_grad():
            embedding = self(image_input)
            
        return embedding