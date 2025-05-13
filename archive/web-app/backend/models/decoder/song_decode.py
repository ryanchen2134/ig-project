import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

# Song Feature Encoder
class SongEncoder(nn.Module):
    def __init__(self, input_dim, embedding_dim=128):
        super(SongEncoder, self).__init__()
        # Multi-layer projection for song features
        self.projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Linear(256, embedding_dim)
        )
        
    def forward(self, x):
        embedding = self.projection(x)
        # Normalize embeddings to lie on unit hypersphere
        return F.normalize(embedding, p=2, dim=1)
