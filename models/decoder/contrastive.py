# models/decoder/contrastive.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from efficientnet_pytorch import EfficientNet
from .img_decoder import ImageEncoder

# Combined model for contrastive learning
class ContrastiveImageSongModel(nn.Module):
    """
    Enhanced contrastive learning model with improved architecture.
    """
    def __init__(self, song_embedding_dim, embedding_dim=256, backbone_type='efficientnet_b3', dropout=0.3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.backbone_type = backbone_type
        
        # Image encoder backbone
        if 'efficientnet' in backbone_type:
            if backbone_type == 'efficientnet_b0':
                self.image_encoder = EfficientNet.from_pretrained('efficientnet-b0')
                in_features = 1280
            elif backbone_type == 'efficientnet_b3':
                self.image_encoder = EfficientNet.from_pretrained('efficientnet-b3')
                in_features = 1536
        elif 'resnet' in backbone_type:
            if backbone_type == 'resnet50':
                self.image_encoder = models.resnet50(pretrained=True)
                in_features = 2048
            else:
                self.image_encoder = models.resnet18(pretrained=True)
                in_features = 512
        else:
            # Default to EfficientNet B0
            self.image_encoder = EfficientNet.from_pretrained('efficientnet-b0')
            in_features = 1280
        
        # Non-linear projection head for images (MLP with BN and ReLU)
        self.image_projector = nn.Sequential(
            nn.Linear(in_features, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
        
        # Song projector (MLP with BN and ReLU)
        self.song_projector = nn.Sequential(
            nn.Linear(song_embedding_dim, embedding_dim * 2),
            nn.BatchNorm1d(embedding_dim * 2),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )
    
    def forward(self, images, song_embeddings=None, image_only=False, song_only=False):
        """
        Forward pass through the model
        
        Args:
            images: Image tensors or None if song_only=True
            song_embeddings: Song feature tensors or None if image_only=True
            image_only: If True, only process images
            song_only: If True, only process songs
        
        Returns:
            Tuple of normalized embeddings depending on inputs
        """
        image_embeddings = None
        projected_song_embeddings = None
        
        # Process images if provided or if image_only mode
        if images is not None or image_only:
            if song_only:
                # Skip image processing in song_only mode
                pass
            else:
                # Extract image features
                if 'efficientnet' in self.backbone_type:
                    image_features = self.image_encoder.extract_features(images)
                    image_features = F.adaptive_avg_pool2d(image_features, (1, 1))
                    image_features = torch.flatten(image_features, 1)
                else:  # ResNet
                    image_features = self.image_encoder.conv1(images)
                    image_features = self.image_encoder.bn1(image_features)
                    image_features = self.image_encoder.relu(image_features)
                    image_features = self.image_encoder.maxpool(image_features)
                    
                    image_features = self.image_encoder.layer1(image_features)
                    image_features = self.image_encoder.layer2(image_features)
                    image_features = self.image_encoder.layer3(image_features)
                    image_features = self.image_encoder.layer4(image_features)
                    
                    image_features = self.image_encoder.avgpool(image_features)
                    image_features = torch.flatten(image_features, 1)
                
                # Project image features
                image_embeddings = self.image_projector(image_features)
                
                # Normalize embeddings to lie on unit hypersphere
                image_embeddings = F.normalize(image_embeddings, p=2, dim=1)
        
        # Process song embeddings if provided or if song_only mode
        if song_embeddings is not None or song_only:
            if image_only:
                # Skip song processing in image_only mode
                pass
            else:
                projected_song_embeddings = self.song_projector(song_embeddings)
                projected_song_embeddings = F.normalize(projected_song_embeddings, p=2, dim=1)
        
        # Return based on mode but always maintain compatibility with expected interfaces
        if image_only:
            return image_embeddings, None
        elif song_only:
            return None, projected_song_embeddings
        else:
            return image_embeddings, projected_song_embeddings
    
    def similarity(self, image_embeddings, song_embeddings, temperature=0.05):
        """Compute similarity matrix with temperature scaling"""
        return torch.matmul(image_embeddings, song_embeddings.T) / temperature


# Improved NTXentLoss with hard negative mining for smaller datasets
class NTXentLoss(nn.Module):
    """
    NT-Xent loss with temperature scaling and hard negative mining.
    """
    def __init__(self, temperature=0.05, hard_negative_weight=0.5, use_hard_negatives=True):
        super().__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.use_hard_negatives = use_hard_negatives
        self.criterion = nn.CrossEntropyLoss(reduction="sum")
        
    def forward(self, image_embeddings, song_embeddings, temperature=None):
        # Use provided temperature or default
        temp = temperature if temperature is not None else self.temperature
        
        # Compute similarity matrix
        sim_matrix = torch.matmul(image_embeddings, song_embeddings.T) / temp
        
        # Get batch size
        batch_size = image_embeddings.shape[0]
        
        # Labels are the diagonal indices (matched pairs)
        labels = torch.arange(batch_size, device=image_embeddings.device)
        
        # If using hard negatives, modify the similarity matrix
        if self.use_hard_negatives and self.hard_negative_weight > 0:
            # Get hardest negative for each sample (highest non-diagonal value)
            with torch.no_grad():
                # Create mask to ignore diagonal (positive) pairs
                mask = torch.eye(batch_size, device=image_embeddings.device) == 0
                
                # Get hardest negatives
                hardest_negatives_i2s = (sim_matrix * mask).max(dim=1)[1]
                hardest_negatives_s2i = (sim_matrix.t() * mask).max(dim=1)[1]
                
            # Blend in hard negatives with regular similarity matrix
            sim_matrix_with_hard_i2s = sim_matrix.clone()
            sim_matrix_with_hard_s2i = sim_matrix.t().clone()
            
            # Increase the similarity of hard negatives
            for i in range(batch_size):
                hn_idx = hardest_negatives_i2s[i]
                sim_matrix_with_hard_i2s[i, hn_idx] = sim_matrix[i, hn_idx] * (1 + self.hard_negative_weight)
                
                hn_idx = hardest_negatives_s2i[i]
                sim_matrix_with_hard_s2i[i, hn_idx] = sim_matrix.t()[i, hn_idx] * (1 + self.hard_negative_weight)
                
            # Compute losses with hard negatives
            loss_i2s = self.criterion(sim_matrix_with_hard_i2s, labels) / batch_size
            loss_s2i = self.criterion(sim_matrix_with_hard_s2i, labels) / batch_size
        else:
            # Standard contrastive loss
            loss_i2s = self.criterion(sim_matrix, labels) / batch_size
            loss_s2i = self.criterion(sim_matrix.t(), labels) / batch_size
        
        # Average both directions
        total_loss = (loss_i2s + loss_s2i) / 2
        
        return total_loss