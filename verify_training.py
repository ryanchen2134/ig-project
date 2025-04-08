# verify_training.py
import torch
import matplotlib.pyplot as plt
import numpy as np
import time
import os
import logging
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
import torch.nn.functional as F
from sklearn.manifold import TSNE

# Import your components
from models.decoder.contrastive import ContrastiveImageSongModel, NTXentLoss
from data.dloader import ImgSongDataset
from train import train_contrastive_model  # Import your training function

# Set paths and hyperparameters
data_path = "DISCO/song_csv/Lady Gaga-Die with a smile.csv"
img_folder = "models/decoder/test_img"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def mini_model():
    # Define parameters
    embedding_dim = 128
    learning_rate = 1e-4
    temperature = 0.07
    
    # Create dataset
    dataset = ImgSongDataset(data_path=data_path, img_folder=img_folder)
    print(f"Dataset size: {len(dataset)}")
    
    # Get song feature dimension
    _, first_song_features = dataset[0]
    song_feature_dim = first_song_features.shape[0]
    print(f"Song feature dimension: {song_feature_dim}")
    
    # Create loss function
    loss_fn = NTXentLoss(temperature=temperature, batch_size=10)
    
    # Create a tiny subset
    mini_indices = np.random.choice(len(dataset), size=20, replace=False)
    mini_dataset = torch.utils.data.Subset(dataset, mini_indices)
    mini_loader = DataLoader(mini_dataset, batch_size=10, shuffle=True)

    # Set up minimal training
    mini_model = ContrastiveImageSongModel(song_feature_dim=song_feature_dim, embedding_dim=embedding_dim).to(device)
    mini_optimizer = optim.Adam(mini_model.parameters(), lr=learning_rate)

    # Run for just a few iterations
    mini_model.train()
    for epoch in range(3):
        for images, song_features in mini_loader:
            images, song_features = images.to(device), song_features.to(device)
            mini_optimizer.zero_grad()
            
            img_emb, song_emb = mini_model(images, song_features)
            loss = loss_fn(img_emb, song_emb)
            
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            # Verify loss is decreasing across iterations
            loss.backward()
            mini_optimizer.step()


# Verification steps as functions
#def verify_data_loading():
  #  print("\n=== VERIFYING DATA LOADING ===")
    # Your data loading verification code here
    # ...

#def verify_minimal_training():
 #   print("\n=== VERIFYING MINIMAL TRAINING ===")
    # Your minimal training verification code here
    # ...

# Add more verification functions...

# Main verification sequence
if __name__ == "__main__":
    mini_model()
"""
    print(f"Running verification steps using device: {device}")
    
    # Run steps in sequence
    verify_data_loading()
    verify_minimal_training()
    # Call other verification functions...
    
    print("\nAll verification steps completed!")
    """