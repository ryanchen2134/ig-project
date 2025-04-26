# models/decoder/contrastive.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .img_decoder import ImageEncoder

# Combined model for contrastive learning
class ContrastiveImageSongModel(nn.Module):
    def __init__(self, song_embedding_dim, embedding_dim=64, backbone_type='resnet18'):
        """
        Initialize the contrastive model for image-song matching.
        
        Args:
            song_embedding_dim: Dimension of the input song embeddings
            embedding_dim: Dimension of the shared embedding space (reduced for smaller dataset)
            backbone_type: Type of image encoder backbone ('resnet18', 'efficientnet_b0', or 'convnext_tiny')
        """
        super(ContrastiveImageSongModel, self).__init__()
        
        # Use the modified ImageEncoder with selected backbone
        self.image_encoder = ImageEncoder(embedding_dim=embedding_dim, backbone_type=backbone_type)
        
        # Song projection with more regularization for smaller datasets
        self.song_projection = nn.Sequential(
            nn.Linear(song_embedding_dim, 256),  # Reduced intermediate dimension
            nn.BatchNorm1d(256),
            nn.GELU(),  
            nn.Dropout(p=0.5),  # Increased dropout to prevent overfitting
            nn.Linear(256, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        # Weight initialization for better training on small datasets
        self._initialize_weights()

        self.embedding_dim = embedding_dim
        
    def _initialize_weights(self):
        """Initialize the weights of the song projection layers for better convergence"""
        for m in self.song_projection.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        
    def forward(self, images, song_embeddings):
        """Forward pass through the model"""
        image_embeddings = self.image_encoder(images)
        projected_song_embeddings = F.normalize(self.song_projection(song_embeddings), p=2, dim=1)
        return image_embeddings, projected_song_embeddings
        
    def get_image_embedding(self, image):
        """Get the embedding for an image"""
        return self.image_encoder(image)


# Improved NTXentLoss with hard negative mining for smaller datasets
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.1, hard_negative_weight=0.5):
        """
        Initialize the NT-Xent loss function with hard negative mining.
        
        Args:
            temperature: Temperature parameter to scale the similarity scores
            hard_negative_weight: Weight for hard negatives mining (0.0 to disable)
        """
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, image_embeddings, song_embeddings):
        """Compute the NT-Xent loss with optional hard negative mining"""
        # Compute similarity matrix
        similarity_matrix = torch.matmul(image_embeddings, song_embeddings.T) / self.temperature
        
        # For InfoNCE loss, the positive samples are the diagonal elements
        labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
        
        # Apply hard negative mining if enabled
        if self.hard_negative_weight > 0:
            # Find hard negatives (high similarity but incorrect matches)
            similarity_matrix_detached = similarity_matrix.detach().clone()
            
            # Zero out the diagonal (positive pairs)
            mask = torch.eye(similarity_matrix_detached.shape[0], dtype=torch.bool, 
                            device=similarity_matrix_detached.device)
            similarity_matrix_detached.masked_fill_(mask, float('-inf'))
            
            # Get indices of hardest negatives
            hardest_negatives = similarity_matrix_detached.max(dim=1)[1]
            
            # Create logits for the standard and hard negative cases
            standard_logits = similarity_matrix
            
            # Create hard negative logits by boosting the hardest negatives
            hard_negative_logits = similarity_matrix.clone()
            for i in range(hard_negative_logits.shape[0]):
                hard_negative_logits[i, hardest_negatives[i]] *= (1 + self.hard_negative_weight)
            
            # Compute loss for both directions with hard negatives
            loss_i2s = self.criterion(hard_negative_logits, labels)
            loss_s2i = self.criterion(hard_negative_logits.T, labels)
        else:
            # Standard NT-Xent loss without hard negative mining
            loss_i2s = self.criterion(similarity_matrix, labels)
            loss_s2i = self.criterion(similarity_matrix.T, labels)
        
        # Total loss is the average of both directions
        total_loss = (loss_i2s + loss_s2i) / 2
        
        return total_loss