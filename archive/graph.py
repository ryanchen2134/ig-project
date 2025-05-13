import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import umap
import torch
import seaborn as sns
import os

from matplotlib.colors import ListedColormap

def visualize_embeddings(image_embeddings, song_embeddings, method='both', n_components=2, 
                         perplexity=30, n_neighbors=15, min_dist=0.1, epoch=None, save_dir='visualizations'):
    """
    Visualize image and song embeddings using dimensionality reduction.
    
    Args:
        image_embeddings: Tensor of image embeddings
        song_embeddings: Tensor of song embeddings
        method: 'tsne', 'umap', or 'both'
        n_components: Dimensionality of the embedding
        perplexity: t-SNE perplexity parameter
        n_neighbors: UMAP n_neighbors parameter
        min_dist: UMAP min_dist parameter
        epoch: Current training epoch (for naming the saved file)
        save_dir: Directory to save visualizations
    """
    # Create directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Convert to numpy arrays if tensors
    if isinstance(image_embeddings, torch.Tensor):
        image_embeddings = image_embeddings.detach().cpu().numpy()
    if isinstance(song_embeddings, torch.Tensor):
        song_embeddings = song_embeddings.detach().cpu().numpy()
    
    # Combine embeddings and create labels
    all_embeddings = np.vstack([image_embeddings, song_embeddings])
    embedding_labels = np.array(['Image'] * len(image_embeddings) + ['Song'] * len(song_embeddings))
    
    # Create matching pairs labels (for coloring)
    match_labels = np.concatenate([np.arange(len(image_embeddings)), np.arange(len(song_embeddings))])
    
    # Set up the figure
    if method == 'both':
        fig, axes = plt.subplots(1, 2, figsize=(20, 8))
    else:
        fig, ax = plt.subplots(figsize=(10, 8))
        axes = [ax]
    
    # Set color palettes
    modal_cmap = ListedColormap(['#1f77b4', '#ff7f0e'])  # Blue for images, orange for songs
    
    # Create pair-matching colors (different color for each pair)
    pair_colors = plt.cm.rainbow(np.linspace(0, 1, len(image_embeddings)))
    
    # Add title info
    title_suffix = f" (Epoch {epoch})" if epoch is not None else ""
    
    # Plot t-SNE if requested
    if method == 'tsne' or method == 'both':
        ax_idx = 0 if method == 'both' else 0
        
        # Apply t-SNE
        tsne = TSNE(n_components=n_components, perplexity=perplexity, n_jobs=-1, random_state=42)
        tsne_result = tsne.fit_transform(all_embeddings)
        
        # Plot by modality (image vs song)
        scatter = axes[ax_idx].scatter(tsne_result[:, 0], tsne_result[:, 1], 
                     c=np.where(embedding_labels == 'Image', 0, 1), 
                     cmap=modal_cmap, alpha=0.7, s=50)
        
        # Draw lines between matching pairs
        for i in range(len(image_embeddings)):
            image_idx = i
            song_idx = len(image_embeddings) + i
            axes[ax_idx].plot([tsne_result[image_idx, 0], tsne_result[song_idx, 0]],
                   [tsne_result[image_idx, 1], tsne_result[song_idx, 1]],
                   color=pair_colors[i], alpha=0.3)
        
        axes[ax_idx].set_title(f"t-SNE Visualization{title_suffix}")
        axes[ax_idx].legend(*scatter.legend_elements(), title="Modality", loc="best")
        axes[ax_idx].grid(alpha=0.3)
    
    # Plot UMAP if requested
    if method == 'umap' or method == 'both':
        ax_idx = 1 if method == 'both' else 0
        
        # Apply UMAP
        reducer = umap.UMAP(n_components=n_components, n_neighbors=n_neighbors, 
                           min_dist=min_dist, random_state=42)
        umap_result = reducer.fit_transform(all_embeddings)
        
        # Plot by modality (image vs song)
        scatter = axes[ax_idx].scatter(umap_result[:, 0], umap_result[:, 1], 
                     c=np.where(embedding_labels == 'Image', 0, 1), 
                     cmap=modal_cmap, alpha=0.7, s=50)
        
        # Draw lines between matching pairs
        for i in range(len(image_embeddings)):
            image_idx = i
            song_idx = len(image_embeddings) + i
            axes[ax_idx].plot([umap_result[image_idx, 0], umap_result[song_idx, 0]],
                   [umap_result[image_idx, 1], umap_result[song_idx, 1]],
                   color=pair_colors[i], alpha=0.3)
        
        axes[ax_idx].set_title(f"UMAP Visualization{title_suffix}")
        axes[ax_idx].legend(*scatter.legend_elements(), title="Modality", loc="best")
        axes[ax_idx].grid(alpha=0.3)
    
    plt.tight_layout()
    
    # Save figure
    epoch_str = f"_epoch_{epoch}" if epoch is not None else ""
    save_path = os.path.join(save_dir, f"embedding_viz_{method}{epoch_str}.png")
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Visualization saved to {save_path}")
    
    # Optionally close the figure to save memory during training
    plt.close(fig)