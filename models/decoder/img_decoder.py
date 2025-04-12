import torch
import torch.nn as nn
import torchvision.models as models
import torch.nn.functional as F

class ImageEncoder(nn.Module):
    def __init__(self, embedding_dim=128):
        super(ImageEncoder, self).__init__()
        # Loading pretrained ConvNeXt model
        convnext = models.convnext_tiny(weights=models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        
        # Remove Classification Layer
        self.backbone = convnext.features

        # Feature Dim
        self.feature_dim = 768

        # Projection Layer
        self.projection = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(self.feature_dim, 512),
            nn.BatchNorm1d(embedding_dim),
            nn.GELU(),
            nn.Linear(512, embedding_dim)
        )  

    def forward(self, x):
        features = self.backbone(x)
        embedding = self.projection(features)
        # Normalize embeddings to lie on unit hypersphere
        return F.normalize(embedding, p=2, dim=1)


idec = ImageEncoder()
if __name__ == "__main__":
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    dummy_input = torch.randn(1, 3, 224, 224)  # Example: batch size of 1, 3 color channels, 224x224 image
    # Pass the dummy input through the encoder
    output = idec(dummy_input)
    # Print the output shape and tensor
    print("Output shape:", output.shape)
    print("Output tensor:", output)