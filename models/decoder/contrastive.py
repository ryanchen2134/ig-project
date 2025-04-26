# models/decoder/contrastive.py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .song_decode import SongEncoder
from .img_decoder import ImageEncoder

# Combined model for contrastive learning
class ContrastiveImageSongModel(nn.Module):
    def __init__(self, song_embedding_dim, embedding_dim=128):
        super(ContrastiveImageSongModel, self).__init__()
        self.image_encoder = ImageEncoder(embedding_dim)
        #self.song_encoder = SongEncoder(song_feature_dim, embedding_dim)

        self.song_projection = nn.Sequential(
            nn.Linear(song_embedding_dim, 512),  # Add intermediate layer
            nn.BatchNorm1d(512),
            nn.GELU(),  
            nn.Dropout(p=0.4),  
            nn.Linear(512, embedding_dim),
            nn.BatchNorm1d(embedding_dim)
        )

        self.embedding_dim = embedding_dim
        
    def forward(self, images, song_embeddings):
        image_embeddings = self.image_encoder(images)
        #song_embeddings = self.song_encoder(song_features)
        projected_song_embeddings = F.normalize(self.song_projection(song_embeddings), p=2, dim=1)
        return image_embeddings, projected_song_embeddings
        
    def get_image_embedding(self, image):
        return self.image_encoder(image)
        
    def get_song_embedding(self, song_feature):
        return self.song_encoder(song_feature)

# InfoNCE/NT-Xent loss for contrastive learning
class NTXentLoss(nn.Module):
    def __init__(self, temperature=0.5, batch_size=32):
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.batch_size = batch_size
        self.criterion = nn.CrossEntropyLoss()
        
    def forward(self, image_embeddings, song_embeddings):
        # Compute similarity matrix
        similarity_matrix = torch.matmul(image_embeddings, song_embeddings.T) / self.temperature
        
        # For InfoNCE loss, the positive samples are the diagonal elements
        labels = torch.arange(similarity_matrix.size(0)).to(similarity_matrix.device)
        
        # Compute loss in both directions
        loss_i2s = self.criterion(similarity_matrix, labels)
        loss_s2i = self.criterion(similarity_matrix.T, labels)
        
        # Total loss is the average of both directions
        total_loss = (loss_i2s + loss_s2i) / 2
        
        return total_loss