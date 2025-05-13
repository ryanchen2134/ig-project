import torch
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import os
from typing import Tuple, Optional, Literal

def visualize_embeddings(image_embeddings: torch.Tensor, 
                         song_embeddings: torch.Tensor,
                         method: Literal['tsne', 'pca', 'both'] = 'tsne', 
                         epoch: Optional[int] = None,
                         save_dir: Optional[str] = None,
                         n_samples: Optional[int] = None,
                         perplexity: int = 30) -> None:
    """
    Visualize embeddings in 2D using t-SNE or PCA
    
    Args:
        image_embeddings: Image embeddings tensor
        song_embeddings: Song embeddings tensor
        method: Visualization method ('tsne', 'pca', or 'both')
        epoch: Current epoch (for filename)
        save_dir: Directory to save visualization
        n_samples: Number of samples to use (None for all)
        perplexity: Perplexity for t-SNE
    """
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    
    # Move to CPU and convert to numpy
    image_embeddings = image_embeddings.cpu().numpy()
    song_embeddings = song_embeddings.cpu().numpy()
    
    # Use a subset of samples if specified
    if n_samples is not None and n_samples < len(image_embeddings):
        indices = np.random.choice(len(image_embeddings), n_samples, replace=False)
        image_embeddings = image_embeddings[indices]
        song_embeddings = song_embeddings[indices]
    
    # Combine embeddings for visualization
    all_embeddings = np.vstack((image_embeddings, song_embeddings))
    
    # Create labels for coloring
    labels = np.concatenate([
        np.zeros(len(image_embeddings)),
        np.ones(len(song_embeddings))
    ])
    
    # Paired labels for matching images and songs
    pair_labels = np.concatenate([
        np.arange(len(image_embeddings)),
        np.arange(len(song_embeddings))
    ])
    
    # Apply dimensionality reduction
    if method == 'tsne' or method == 'both':
        # Apply t-SNE
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        embeddings_2d_tsne = tsne.fit_transform(all_embeddings)
        
        # Create t-SNE plot
        plt.figure(figsize=(10, 8))
        
        # Plot scatter points
        scatter = plt.scatter(
            embeddings_2d_tsne[:, 0], 
            embeddings_2d_tsne[:, 1], 
            c=labels, 
            cmap='coolwarm', 
            alpha=0.8
        )
        
        # Draw lines between matching pairs
        for i in range(len(image_embeddings)):
            idx1 = i  # Image embedding index
            idx2 = i + len(image_embeddings)  # Matching song embedding index
            plt.plot(
                [embeddings_2d_tsne[idx1, 0], embeddings_2d_tsne[idx2, 0]],
                [embeddings_2d_tsne[idx1, 1], embeddings_2d_tsne[idx2, 1]],
                'k-', alpha=0.3
            )
        
        # Add legend
        legend1 = plt.legend(*scatter.legend_elements(),
                            title="Types")
        plt.gca().add_artist(legend1)
        
        plt.title(f't-SNE Visualization of Embeddings{f" (Epoch {epoch})" if epoch else ""}')
        
        if save_dir:
            filename = f'tsne_epoch_{epoch}.png' if epoch is not None else 'tsne.png'
            plt.savefig(os.path.join(save_dir, filename), dpi=300)
        
        plt.close()
    
    if method == 'pca' or method == 'both':
        # Apply PCA
        pca = PCA(n_components=2)
        embeddings_2d_pca = pca.fit_transform(all_embeddings)
        
        # Create PCA plot
        plt.figure(figsize=(10, 8))
        
        # Plot scatter points
        scatter = plt.scatter(
            embeddings_2d_pca[:, 0], 
            embeddings_2d_pca[:, 1], 
            c=labels, 
            cmap='coolwarm', 
            alpha=0.8
        )
        
        # Draw lines between matching pairs
        for i in range(len(image_embeddings)):
            idx1 = i  # Image embedding index
            idx2 = i + len(image_embeddings)  # Matching song embedding index
            plt.plot(
                [embeddings_2d_pca[idx1, 0], embeddings_2d_pca[idx2, 0]],
                [embeddings_2d_pca[idx1, 1], embeddings_2d_pca[idx2, 1]],
                'k-', alpha=0.3
            )
        
        # Add legend
        legend1 = plt.legend(*scatter.legend_elements(),
                            title="Types")
        plt.gca().add_artist(legend1)
        
        plt.title(f'PCA Visualization of Embeddings{f" (Epoch {epoch})" if epoch else ""}')
        
        if save_dir:
            filename = f'pca_epoch_{epoch}.png' if epoch is not None else 'pca.png'
            plt.savefig(os.path.join(save_dir, filename), dpi=300)
        
        plt.close()