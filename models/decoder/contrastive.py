# models/decoder/contrastive.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from .img_decoder import ImageEncoder

# Combined model for contrastive learning
class ContrastiveImageSongModel(nn.Module):
    def __init__(self, song_embedding_dim, embedding_dim=128, backbone_type='resnet18', projection_dim=512):
        """
        Enhanced contrastive model with additional regularization and more expressive projections
        
        Args:
            song_embedding_dim: Dimension of the input song embeddings
            embedding_dim: Dimension of the shared embedding space
            backbone_type: Type of image encoder backbone
            projection_dim: Intermediate projection dimension
        """
        super(ContrastiveImageSongModel, self).__init__()
        
        # Use the modified ImageEncoder with selected backbone
        self.image_encoder = ImageEncoder(embedding_dim=embedding_dim, backbone_type=backbone_type)
        
        # More expressive song projection 
        self.song_projection = nn.Sequential(
            nn.Linear(song_embedding_dim, projection_dim),
            nn.LayerNorm(projection_dim),  # LayerNorm works better than GroupNorm for small batches
            nn.GELU(),
            nn.Dropout(p=0.5),
            nn.Linear(projection_dim, projection_dim // 2),
            nn.LayerNorm(projection_dim // 2),
            nn.GELU(),
            nn.Dropout(p=0.3),
            nn.Linear(projection_dim // 2, embedding_dim)
        )

        # Add similarity scaling parameter (learnable temperature)
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
        
        self._initialize_weights()
        self.embedding_dim = embedding_dim
        
    def _initialize_weights(self):
        """Better weight initialization for deep projections"""
        for m in self.song_projection.modules():
            if isinstance(m, nn.Linear):
                # Use better initialization for GELU
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        
    def forward(self, images=None, song_features=None, song_only=False, image_only=False):
        """Forward pass with automatic scaling of similarities"""
        if song_only and song_features is not None:
            song_embedding = self.song_projection(song_features)
            song_embedding = F.normalize(song_embedding, p=2, dim=1)
            return None, song_embedding
            
        if image_only and images is not None:
            image_embedding = self.image_encoder(images)
            image_embedding = F.normalize(image_embedding, p=2, dim=1)
            return image_embedding, None
        
        # Process both modalities
        image_embedding = self.image_encoder(images)
        song_embedding = self.song_projection(song_features)
        
        # Normalize embeddings
        image_embedding = F.normalize(image_embedding, p=2, dim=1)
        song_embedding = F.normalize(song_embedding, p=2, dim=1)
        
        return image_embedding, song_embedding
    
    def similarity(self, image_embeddings, song_embeddings):
        """Get scaled cosine similarity with learned temperature"""
        # Clamp for stability
        scale = torch.clamp(self.logit_scale.exp(), min=1.0, max=100.0)
        return scale * torch.matmul(image_embeddings, song_embeddings.T)



# Improved NTXentLoss with hard negative mining for smaller datasets
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1, hard_negative_weight=0.3, hardest_only=False):
        """        
        Args:
            temperature: Temperature parameter (not used if model has learned scaling)
            hard_negative_weight: Weight for hard negatives mining
            hardest_only: If True, only consider the hardest negative
        """
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.hardest_only = hardest_only
        self.criterion = nn.CrossEntropyLoss()
        self.margin = 0.3  
        
    def forward(self, image_embeddings, song_embeddings, use_model_temp=True, model_sim=None):
        batch_size = image_embeddings.size(0)
        
        # Use provided similarity or compute it
        if model_sim is not None:
            similarity_matrix = model_sim
        else:
            similarity_matrix = torch.matmul(image_embeddings, song_embeddings.T)
            if not use_model_temp:
                similarity_matrix = similarity_matrix / self.temperature
        
        # For InfoNCE loss, the positive samples are the diagonal elements
        labels = torch.arange(batch_size).to(similarity_matrix.device)
        
        # Get positive pair similarities (diagonal)
        pos_sim = torch.diag(similarity_matrix)
        
        if self.hard_negative_weight > 0:
            similarity_matrix_detached = similarity_matrix.detach().clone()
            
            # Create mask for positives
            mask = torch.eye(batch_size, dtype=torch.bool, device=similarity_matrix_detached.device)
            similarity_matrix_detached.masked_fill_(mask, float('-inf'))
            
            # Find hard negatives (highest similarity incorrect matches)
            hard_negatives_values, hard_negatives = torch.topk(similarity_matrix_detached, 
                                                              k=2 if not self.hardest_only else 1, 
                                                              dim=1)
            
            # Create boosted similarity matrix
            boosted_sim = similarity_matrix.clone()
            
            # Apply weighting to the hardest negatives
            for i in range(boosted_sim.shape[0]):
                for j, neg_idx in enumerate(hard_negatives[i]):
                    # Weight decreases as we move from hardest to less hard
                    weight = self.hard_negative_weight / (j + 1) if not self.hardest_only else self.hard_negative_weight
                    boosted_sim[i, neg_idx] *= (1 + weight)
            
            # Standard InfoNCE losses with hard negatives
            loss_i2s = self.criterion(boosted_sim, labels)
            loss_s2i = self.criterion(boosted_sim.T, labels)
            
            # Add margin-based contrastive component for most difficult negatives
            neg_sim = hard_negatives_values[:, 0]  
            margin_loss = torch.mean(torch.clamp(neg_sim - pos_sim + self.margin, min=0))
            
            # Combined loss
            total_loss = (loss_i2s + loss_s2i) / 2 + 0.3 * margin_loss
            
        else:
            # Standard NT-Xent loss 
            loss_i2s = self.criterion(similarity_matrix, labels)
            loss_s2i = self.criterion(similarity_matrix.T, labels)
            total_loss = (loss_i2s + loss_s2i) / 2
        
        return total_loss