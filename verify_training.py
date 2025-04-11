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
from train import train_contrastive_model 

# Set paths and hyperparameters
data_path = "data/csv files/pairs_songencoded(1).csv"
img_folder = "models/decoder/test_img"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#Dataloader test - Worked but need to fix song_embeds to be size: 128
def data_load():
    # Example usage
    dataset = ImgSongDataset(data_path, img_folder)
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # Test loading a sample
    sample_img, sample_features = dataset[0]
    print(f"Image shape: {sample_img.shape}")
    print(f"Features shape: {sample_features.shape}")

#Test no. 1
def mini_model():
    # Define parameters
    print("\n=== MINI MODEL TEST ===")
    embedding_dim = 64
    learning_rate = 1e-5
    temperature = 0.07
    
    # Create dataset
    dataset = ImgSongDataset(file_path=data_path, img_folder=img_folder)
    print(f"Dataset size: {len(dataset)}")
    
    # Get song feature dimension
    _, first_song_features = dataset[0]
    song_feature_dim = first_song_features.shape[0]
    print(f"Song feature dimension: {song_feature_dim}")
    
    # Create loss function
    batch_size = min(10, len(dataset))
    loss_fn = NTXentLoss(temperature=temperature, batch_size=batch_size)
    
    # Create a tiny subset
    subset_size = min(20, len(dataset))
    mini_indices = np.random.choice(len(dataset), size=subset_size, replace=False)
    mini_dataset = torch.utils.data.Subset(dataset, mini_indices)
    mini_loader = DataLoader(mini_dataset, batch_size=10, shuffle=True)

    # Set up minimal training
    mini_model = ContrastiveImageSongModel(song_embedding_dim=song_feature_dim, embedding_dim=embedding_dim).to(device)
    mini_optimizer = optim.Adam(mini_model.parameters(), lr=learning_rate)

    # Run for just a few iterations
    mini_model.train()
    for epoch in range(10):
        for images, song_features in mini_loader:
            images, song_features = images.to(device), song_features.to(device)
            mini_optimizer.zero_grad()
            
            img_emb, song_emb = mini_model(images, song_features)
            loss = loss_fn(img_emb, song_emb)
            
            print(f"Epoch {epoch}, Loss: {loss.item():.4f}")
            
            # Verify loss is decreasing across iterations
            loss.backward()
            mini_optimizer.step()

#Test no. 2
def verify_data_loading():
    print("\n=== VERIFYING DATA LOADING ===")
    # Get a single batch
    images, song_features = next(iter(train_loader))

    # Check shapes
    print(f"Images batch shape: {images.shape}")  # Should be [batch_size, channels, height, width]
    print(f"Song features shape: {song_features.shape}")  # Should be [batch_size, feature_dim]

    # Check value ranges
    print(f"Image min/max: {images.min().item():.4f}/{images.max().item():.4f}")
    print(f"Song features min/max: {song_features.min().item():.4f}/{song_features.max().item():.4f}")

    # Visualize images (first 4 in batch)
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for i in range(4):
        # Convert tensor to numpy and transpose for matplotlib
        img = images[i].cpu().numpy().transpose(1, 2, 0)
        # Denormalize if needed (assuming normalization with mean 0.5, std 0.5)
        img = img * 0.5 + 0.5
        axes[i].imshow(img)
        axes[i].axis('off')
    plt.show()

    # Check song features statistics
    print("Song features statistics:")
    print(f"Mean: {song_features.mean(dim=0)[:5]}")  # First 5 feature means
    print(f"Std: {song_features.std(dim=0)[:5]}")  # First 5 feature std devs
    print(f"Any NaN: {torch.isnan(song_features).any()}")
    print(f"Any Inf: {torch.isinf(song_features).any()}")

#Test no. 3
def validate_embeddings():
    # Get a batch
    print("\n=== VALIDATING EMBEDDINGS ===")
    images, song_features = next(iter(train_loader))
    images, song_features = images.to(device), song_features.to(device)

    # Forward pass
    model.eval()
    with torch.no_grad():
        img_emb, song_emb = model(images, song_features)

    # Check embedding shapes
    print(f"Image embedding shape: {img_emb.shape}")  # Should be [batch_size, embedding_dim]
    print(f"Song embedding shape: {song_emb.shape}")  # Should be [batch_size, embedding_dim]

    # Check normalization (L2 norm should be close to 1)
    img_norms = torch.norm(img_emb, dim=1)
    song_norms = torch.norm(song_emb, dim=1)
    print(f"Image embedding norms: mean={img_norms.mean().item():.4f}, std={img_norms.std().item():.4f}")
    print(f"Song embedding norms: mean={song_norms.mean().item():.4f}, std={song_norms.std().item():.4f}")

    # Check pair similarities vs. random similarities
    pair_sims = torch.sum(img_emb * song_emb, dim=1)  # Dot product of matched pairs
    random_sims = []
    for i in range(len(img_emb)):
        # Get similarity with a random non-matching song
        j = (i + 1) % len(song_emb)  # Simple way to get a different index
        random_sims.append(torch.sum(img_emb[i] * song_emb[j]).item())

    print(f"Matched pair similarities: {pair_sims.mean().item():.4f} ± {pair_sims.std().item():.4f}")
    print(f"Random pair similarities: {np.mean(random_sims):.4f} ± {np.std(random_sims):.4f}")

#Test no. 4
def test_loss():
    print("\n=== VERIFYING LOSS FUNCTION ===")
    # Create controlled dummy embeddings
    batch_size = 8
    embed_dim = 128

    # Case 1: Identical embeddings (perfect match)
    identical_emb = F.normalize(torch.ones(batch_size, embed_dim), dim=1)
    loss_identical = loss_fn(identical_emb, identical_emb)

    # Case 2: Orthogonal embeddings (no similarity)
    a = F.normalize(torch.ones(batch_size, embed_dim), dim=1)
    b = F.normalize(torch.ones(batch_size, embed_dim), dim=1)
    # Make b orthogonal to a
    b[:, :embed_dim//2] = -1
    loss_orthogonal = loss_fn(a, b)

    # Case 3: Mixed similarity (partially similar)
    c = F.normalize(torch.randn(batch_size, embed_dim), dim=1)
    d = F.normalize(torch.randn(batch_size, embed_dim), dim=1)
    # Make diagonal pairs more similar
    for i in range(batch_size):
        d[i] = c[i] * 0.8 + d[i] * 0.2
    d = F.normalize(d, dim=1)
    loss_mixed = loss_fn(c, d)

    print(f"Loss for identical embeddings: {loss_identical.item():.4f}")
    print(f"Loss for orthogonal embeddings: {loss_orthogonal.item():.4f}")
    print(f"Loss for mixed similarity: {loss_mixed.item():.4f}")

#Test no. 5
def run_partial_training(portion=0.1, epochs=5):
    # Create a subset
    subset_size = int(len(dataset) * portion)
    indices = np.random.choice(len(dataset), size=subset_size, replace=False)
    subset = torch.utils.data.Subset(dataset, indices)
    subset_loader = DataLoader(subset, batch_size=batch_size, shuffle=True, num_workers=2)
    
    # Create fresh model and optimizer
    subset_model = ContrastiveImageSongModel(song_feature_dim=song_feature_dim, 
                                            embedding_dim=embedding_dim).to(device)
    subset_optimizer = optim.Adam(subset_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Train
    start_time = time.time()
    subset_model, history = train_contrastive_model(
        subset_model, subset_loader, val_loader, subset_optimizer, loss_fn,
        num_epochs=epochs, patience=epochs, device=device
    )
    train_time = time.time() - start_time
    
    # Check memory usage
    if torch.cuda.is_available():
        memory_used = torch.cuda.max_memory_allocated(device) / (1024**3)  # GB
    else:
        memory_used = "N/A (CPU)"
    
    print(f"Training with {portion*100}% data ({subset_size} samples):")
    print(f"  Time taken: {train_time:.2f} seconds")
    print(f"  Max memory used: {memory_used} GB")
    print(f"  Final train loss: {history['train_loss'][-1]:.4f}")
    print(f"  Final val loss: {history['val_loss'][-1]:.4f}")
    
    return subset_model, history

#Test no. 6
def plot_overfitting_check(history):
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training vs Validation Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Calculate gap between train and val loss
    gaps = np.array(history['val_loss']) - np.array(history['train_loss'])
    print(f"Initial train/val gap: {gaps[0]:.4f}")
    print(f"Final train/val gap: {gaps[-1]:.4f}")
    if gaps[-1] > 2 * gaps[0]:
        print("WARNING: Gap between train and validation loss is widening significantly")
    
    plt.show()


# Main verification sequence
if __name__ == "__main__":
    # data_load()
    mini_model()

#TEST NO 5
    # Run with increasing portions#
    # model_10pct, history_10pct = run_partial_training(portion=0.1, epochs=5)
    # model_25pct, history_25pct = run_partial_training(portion=0.25, epochs=5)

#TEST no 6
"""
# Check training history for signs of overfitting
plot_overfitting_check(history_25pct)

# Check embedding diversity
model_25pct.eval()
with torch.no_grad():
    # Get embeddings for validation set
    all_img_embs = []
    all_song_embs = []
    for images, song_features in val_loader:
        images, song_features = images.to(device), song_features.to(device)
        img_emb, song_emb = model_25pct(images, song_features)
        all_img_embs.append(img_emb)
        all_song_embs.append(song_emb)
    
    all_img_embs = torch.cat(all_img_embs, dim=0)
    all_song_embs = torch.cat(all_song_embs, dim=0)

# Calculate cosine similarity matrix for images
img_sim_matrix = torch.mm(all_img_embs, all_img_embs.t())
avg_img_sim = (img_sim_matrix.sum() - torch.trace(img_sim_matrix)) / (img_sim_matrix.numel() - img_sim_matrix.size(0))
print(f"Average similarity between different image embeddings: {avg_img_sim.item():.4f}")
print(f"If this value is too high (> 0.5), embeddings may be collapsing to similar values")
"""

"""
    print(f"Running verification steps using device: {device}")
    
    # Run steps in sequence
    verify_data_loading()
    verify_minimal_training()
    # Call other verification functions...
    
    print("\nAll verification steps completed!")
    """