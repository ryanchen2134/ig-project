import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

class ContrastiveImageSongModel(nn.Module):
    def __init__(self, 
                 song_embedding_dim: int, 
                 embedding_dim: int = 128,
                 image_encoder: Optional[nn.Module] = None,
                 clip_model_name: str = "ViT-B/32",
                 freeze_image_encoder: bool = True):
        """
        Enhanced contrastive model with CLIP-based image encoder
        
        Args:
            song_embedding_dim: Dimension of the input song embeddings
            embedding_dim: Dimension of the shared embedding space
            image_encoder: Optional pre-constructed image encoder 
            clip_model_name: CLIP model variant if image_encoder not provided
            freeze_image_encoder: Whether to freeze the image encoder base
        """
        super(ContrastiveImageSongModel, self).__init__()
        
        # Set up image encoder - either use provided or create new
        if image_encoder is not None:
            self.image_encoder = image_encoder
        else:
            from models.encoders.clip_image_encoder import CLIPImageEncoder
            self.image_encoder = CLIPImageEncoder(
                clip_model_name=clip_model_name,
                embedding_dim=embedding_dim,
                freeze_base=freeze_image_encoder
            )
        
        # Song projection layers - with proper normalization
        self.song_projection = nn.Sequential(
            nn.Linear(song_embedding_dim, 512),
            nn.LayerNorm(512),  # Better than BatchNorm for small batches
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, embedding_dim)
        )

        # Add learned temperature parameter for similarity scaling
        self.logit_scale = nn.Parameter(torch.ones([]) * torch.log(torch.tensor(1/0.07)))
        
        self._initialize_weights()
        self.embedding_dim = embedding_dim
        
    def _initialize_weights(self):
        """Initialize projection layer weights"""
        for m in self.song_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, 
                images: Optional[torch.Tensor] = None, 
                song_features: Optional[torch.Tensor] = None,
                song_only: bool = False, 
                image_only: bool = False) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass with flexible inputs for different modes
        
        Args:
            images: Batch of images (B, C, H, W) or None
            song_features: Batch of song embeddings (B, D) or None
            song_only: Process only song features
            image_only: Process only images
            
        Returns:
            Tuple of (image_embedding, song_embedding), either can be None
        """
        # Process only song features
        if song_only and song_features is not None:
            song_embedding = self.song_projection(song_features)
            song_embedding = F.normalize(song_embedding, p=2, dim=1)
            return None, song_embedding
            
        # Process only images
        if image_only and images is not None:
            image_embedding = self.image_encoder(images)
            # Already normalized in the encoder
            return image_embedding, None
        
        # Process both modalities
        image_embedding = self.image_encoder(images)
        song_embedding = self.song_projection(song_features)
        
        # Normalize song embeddings (image already normalized)
        song_embedding = F.normalize(song_embedding, p=2, dim=1)
        
        return image_embedding, song_embedding
    
    def similarity(self, image_embeddings: torch.Tensor, 
                  song_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Compute similarity with learned temperature scaling
        
        Args:
            image_embeddings: Normalized image embeddings
            song_embeddings: Normalized song embeddings
            
        Returns:
            Scaled cosine similarity matrix
        """
        # Clamp for stability
        scale = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)
        return scale * torch.matmul(image_embeddings, song_embeddings.T)