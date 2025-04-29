import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import torch
import os
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import umap.umap_ as umap
import matplotlib.patches as mpatches
from matplotlib.colors import ListedColormap
import pandas as pd
from typing import List, Tuple

class EnhancedVisualization:
    """Enhanced visualization class for contrastive learning models."""
    
    def __init__(self, save_dir='visualizations/', figsize=(24, 16)):
        """Initialize the visualization class."""
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)
        self.figsize = figsize
        self.cmap = plt.cm.get_cmap('tab10')
    
    def plot_training_history(self, history, save_path=None):
        """Plot enhanced training history with multiple metrics."""
        plt.figure(figsize=self.figsize)
        
        # Define the layout
        n_plots = 3 if 'recalls' in history else 2
        fig, axs = plt.subplots(n_plots, 1, figsize=self.figsize, sharex=True)
        
        # 1. Loss plot
        ax = axs[0]
        ax.plot(history['train_loss'], label='Training Loss', color='#1f77b4', linewidth=2)
        ax.plot(history['val_loss'], label='Validation Loss', color='#ff7f0e', linewidth=2)
        
        if 'best_epoch' in history:
            ax.axvline(x=history['best_epoch']-1, color='red', linestyle='--', 
                       label=f'Best Model (Epoch {history["best_epoch"]})')
            
        ax.set_ylabel('Loss', fontsize=12)
        ax.set_title('Contrastive Learning Loss', fontsize=14, fontweight='bold')
        ax.legend(fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 2. Learning Rate plot
        ax = axs[1]
        if 'learning_rates' in history:
            ax.plot(history['learning_rates'], color='#2ca02c', linewidth=2)
            if 'best_epoch' in history:
                ax.axvline(x=history['best_epoch']-1, color='red', linestyle='--')
                
            ax.set_ylabel('Learning Rate', fontsize=12)
            ax.set_title('Learning Rate Schedule', fontsize=14, fontweight='bold')
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # 3. Recall metrics if available
        if 'recalls' in history:
            ax = axs[2]
            recalls = history['recalls']
            
            # Extract R@1, R@5, R@10
            r1_values = [r.get('r@1_i2s', 0) for r in recalls]
            r5_values = [r.get('r@5_i2s', 0) for r in recalls]
            r10_values = [r.get('r@10_i2s', 0) for r in recalls]
            
            ax.plot(r1_values, label='R@1', color='#d62728', linewidth=2)
            ax.plot(r5_values, label='R@5', color='#9467bd', linewidth=2)
            ax.plot(r10_values, label='R@10', color='#8c564b', linewidth=2)
            
            if 'best_epoch' in history:
                ax.axvline(x=history['best_epoch']-1, color='red', linestyle='--')
                
            ax.set_ylabel('Recall (%)', fontsize=12)
            ax.set_title('Recall Metrics', fontsize=14, fontweight='bold')
            ax.legend(fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
        
        # Shared X-axis label
        axs[-1].set_xlabel('Epochs', fontsize=12)
        
        plt.tight_layout()
        
        if save_path is None:
            save_path = f"{self.save_dir}/enhanced_training_history.png"
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_embedding_space(self, 
                                 image_embeddings, 
                                 song_embeddings,
                                 method='both',
                                 epoch=None,
                                 save_prefix='embedding_viz',
                                 use_matching_lines=True):
        """
        Visualize embeddings using multiple dimensionality reduction techniques.
        
        Args:
            image_embeddings: Tensor of image embeddings
            song_embeddings: Tensor of song embeddings
            method: 'tsne', 'pca', 'umap', or 'both' (for both t-SNE and UMAP)
            epoch: Current epoch number for the filename
            save_prefix: Prefix for the saved file
            use_matching_lines: Whether to draw lines between matching pairs
        """
        # Convert to numpy if tensors
        if isinstance(image_embeddings, torch.Tensor):
            image_embeddings = image_embeddings.detach().cpu().numpy()
        if isinstance(song_embeddings, torch.Tensor):
            song_embeddings = song_embeddings.detach().cpu().numpy()
            
        # Create figure
        if method == 'both':
            fig, axs = plt.subplots(1, 2, figsize=self.figsize)
        else:
            fig, ax = plt.subplots(1, 1, figsize=(self.figsize[0]//2, self.figsize[1]))
            axs = [ax]
        
        # Common visualization code
        def visualize_on_axis(ax, img_proj, song_proj, title):
            # Plot points
            ax.scatter(img_proj[:, 0], img_proj[:, 1], c='blue', alpha=0.7, label='Images')
            ax.scatter(song_proj[:, 0], song_proj[:, 1], c='orange', alpha=0.7, label='Songs')
            
            # Draw lines between matching pairs if requested
            if use_matching_lines:
                for i in range(len(img_proj)):
                    ax.plot([img_proj[i, 0], song_proj[i, 0]], 
                            [img_proj[i, 1], song_proj[i, 1]], 
                            'gray', alpha=0.3, linewidth=0.5)
            
            # Add title and legend
            ax.set_title(title, fontsize=14, fontweight='bold')
            ax.legend(fontsize=12)
            
            # Add grid
            ax.grid(True, linestyle='--', alpha=0.3)
        
        # Combine embeddings for dimensionality reduction
        all_embeddings = np.vstack([image_embeddings, song_embeddings])
        
        # Perform visualizations based on method
        if method in ['tsne', 'both']:
            # t-SNE
            tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)-1))
            embeddings_tsne = tsne.fit_transform(all_embeddings)
            
            # Split back to image and song embeddings
            img_tsne = embeddings_tsne[:len(image_embeddings)]
            song_tsne = embeddings_tsne[len(image_embeddings):]
            
            # Visualize
            ax_idx = 0 if method == 'both' else 0
            visualize_on_axis(axs[ax_idx], img_tsne, song_tsne, f't-SNE Visualization{f" (Epoch {epoch})" if epoch else ""}')
        
        if method in ['umap', 'both']:
            # UMAP
            reducer = umap.UMAP(random_state=42, n_neighbors=min(15, len(all_embeddings)-1))
            embeddings_umap = reducer.fit_transform(all_embeddings)
            
            # Split back to image and song embeddings
            img_umap = embeddings_umap[:len(image_embeddings)]
            song_umap = embeddings_umap[len(image_embeddings):]
            
            # Visualize
            ax_idx = 1 if method == 'both' else 0
            visualize_on_axis(axs[ax_idx], img_umap, song_umap, f'UMAP Visualization{f" (Epoch {epoch})" if epoch else ""}')
        
        if method == 'pca':
            # PCA
            pca = PCA(n_components=2, random_state=42)
            embeddings_pca = pca.fit_transform(all_embeddings)
            
            # Split back to image and song embeddings
            img_pca = embeddings_pca[:len(image_embeddings)]
            song_pca = embeddings_pca[len(image_embeddings):]
            
            # Visualize
            visualize_on_axis(axs[0], img_pca, song_pca, f'PCA Visualization{f" (Epoch {epoch})" if epoch else ""}')
        
        # Save figure
        epoch_suffix = f"_epoch{epoch}" if epoch else ""
        save_path = f"{self.save_dir}/{save_prefix}_{method}{epoch_suffix}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def visualize_similarity_matrix(self, 
                                   image_embeddings, 
                                   song_embeddings,
                                   temperature=1.0,
                                   epoch=None,
                                   save_prefix='similarity_matrix'):
        """
        Visualize the similarity matrix between image and song embeddings.
        
        Args:
            image_embeddings: Tensor of image embeddings
            song_embeddings: Tensor of song embeddings
            temperature: Temperature parameter for scaling similarities
            epoch: Current epoch number for the filename
            save_prefix: Prefix for the saved file
        """
        # Convert to numpy if tensors
        if isinstance(image_embeddings, torch.Tensor):
            image_embeddings = image_embeddings.detach().cpu().numpy()
        if isinstance(song_embeddings, torch.Tensor):
            song_embeddings = song_embeddings.detach().cpu().numpy()
            
        # Calculate similarity matrix
        similarity = np.matmul(image_embeddings, song_embeddings.T) / temperature
        
        # Create figure
        plt.figure(figsize=self.figsize)
        
        # Plot heatmap
        ax = sns.heatmap(
            similarity, 
            cmap='YlGnBu',
            xticklabels=False, 
            yticklabels=False,
            cbar_kws={'label': 'Similarity Score'}
        )
        
        # Mark the diagonal (matching pairs)
        for i in range(min(similarity.shape)):
            plt.scatter(i + 0.5, i + 0.5, c='red', s=10)
        
        # Add title and labels
        plt.title(f'Image-Song Similarity Matrix{f" (Epoch {epoch})" if epoch else ""}', 
                 fontsize=16, fontweight='bold')
        plt.xlabel('Songs', fontsize=14)
        plt.ylabel('Images', fontsize=14)
        
        # Save figure
        epoch_suffix = f"_epoch{epoch}" if epoch else ""
        save_path = f"{self.save_dir}/{save_prefix}{epoch_suffix}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path
    
    def plot_recall_analysis(self, model, test_loader, device, k_values=[1, 5, 10, 50], save_prefix='recall_analysis'):
        """
        Perform detailed recall analysis on the test set.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            device: Device to use for inference
            k_values: List of k values for Recall@k
            save_prefix: Prefix for saved files
        """
        model.eval()
        
        # Collect all embeddings and labels
        all_image_embeddings = []
        all_song_embeddings = []
        
        with torch.no_grad():
            for images, song_embeddings in test_loader:
                images = images.to(device)
                song_embeddings = song_embeddings.to(device)
                
                # Get embeddings
                image_embeddings, projected_song_embeddings = model(images, song_embeddings)
                
                # Collect
                all_image_embeddings.append(image_embeddings.cpu())
                all_song_embeddings.append(projected_song_embeddings.cpu())
        
        # Concatenate all batches
        all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
        all_song_embeddings = torch.cat(all_song_embeddings, dim=0)
        
        # Calculate similarity matrix for all samples
        similarity = torch.matmul(all_image_embeddings, all_song_embeddings.T)
        
        # Calculate recall metrics for both directions
        recalls_i2s = []  # Image to song
        recalls_s2i = []  # Song to image
        
        for k in k_values:
            # Image to song
            _, topk_indices_i2s = similarity.topk(k=min(k, similarity.shape[1]), dim=1)
            targets = torch.arange(similarity.shape[0]).view(-1, 1)
            correct_i2s = torch.any(topk_indices_i2s == targets, dim=1).float().mean().item() * 100
            recalls_i2s.append(correct_i2s)
            
            # Song to image
            _, topk_indices_s2i = similarity.T.topk(k=min(k, similarity.shape[0]), dim=1)
            correct_s2i = torch.any(topk_indices_s2i == targets, dim=1).float().mean().item() * 100
            recalls_s2i.append(correct_s2i)
        
        # Plot recalls
        plt.figure(figsize=(12, 8))
        
        # Plot with wider bars
        bar_width = 0.35
        x = np.arange(len(k_values))
        
        plt.bar(x - bar_width/2, recalls_i2s, bar_width, label='Image → Song', color='#1f77b4')
        plt.bar(x + bar_width/2, recalls_s2i, bar_width, label='Song → Image', color='#ff7f0e')
        
        # Calculate average recalls
        avg_recalls = [(i + s) / 2 for i, s in zip(recalls_i2s, recalls_s2i)]
        
        # Add value labels on top of bars
        for i, v in enumerate(recalls_i2s):
            plt.text(i - bar_width/2, v + 1, f'{v:.1f}%', ha='center', fontsize=10)
            
        for i, v in enumerate(recalls_s2i):
            plt.text(i + bar_width/2, v + 1, f'{v:.1f}%', ha='center', fontsize=10)
        
        # Set x-tick labels to k values
        plt.xticks(x, [f'R@{k}' for k in k_values])
        
        plt.xlabel('Recall@k Metrics', fontsize=12)
        plt.ylabel('Recall (%)', fontsize=12)
        plt.title('Detailed Recall Analysis', fontsize=14, fontweight='bold')
        plt.legend(fontsize=12)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Set y-axis limit with some headroom
        plt.ylim(0, max(max(recalls_i2s), max(recalls_s2i)) * 1.2)
        
        # Save figure
        save_path = f"{self.save_dir}/{save_prefix}.png"
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        # Return the metrics as a dictionary
        metrics = {
            f'r@{k}_i2s': i2s for k, i2s in zip(k_values, recalls_i2s)
        }
        metrics.update({
            f'r@{k}_s2i': s2i for k, s2i in zip(k_values, recalls_s2i)
        })
        metrics.update({
            f'r@{k}_avg': avg for k, avg in zip(k_values, avg_recalls)
        })
        
        return metrics, save_path
    
    def top_k_retrievals_visualization(self, 
                                     model, 
                                     test_loader, 
                                     img_folder,
                                     device, 
                                     k=5, 
                                     num_examples=5,
                                     save_prefix='retrieval_examples'):
        """
        Visualize top-k retrievals for a few examples.
        
        Args:
            model: Trained model
            test_loader: Test data loader
            img_folder: Folder containing images
            device: Device to use for inference
            k: Number of top retrievals to show
            num_examples: Number of example queries to show
            save_prefix: Prefix for saved files
        """
        model.eval()
        
        # Get a batch of test data
        images_list = []
        song_embeddings_list = []
        image_paths_list = []
        
        # Collect all test data
        for batch_idx, (images, song_embeddings) in enumerate(test_loader):
            if batch_idx >= num_examples:
                break
                
            images_list.append(images)
            song_embeddings_list.append(song_embeddings)
            
            # Here we'd normally collect image paths too
            # This is a placeholder - in real implementation, we'd get the actual paths
            image_paths_list.extend([f"image_{batch_idx}_{i}.jpg" for i in range(len(images))])
        
        # Concatenate batches
        all_images = torch.cat(images_list, dim=0)
        all_song_embeddings = torch.cat(song_embeddings_list, dim=0)
        
        # Compute embeddings
        with torch.no_grad():
            all_images = all_images.to(device)
            all_song_embeddings = all_song_embeddings.to(device)
            
            image_embeddings, projected_song_embeddings = model(all_images, all_song_embeddings)
            
        # Compute similarity matrix
        similarity = torch.matmul(image_embeddings, projected_song_embeddings.T)
        
        # For each query image, get top-k retrievals
        k = min(k, similarity.shape[1])
        
        # Get top-k indices for each query
        _, topk_indices = similarity[:num_examples].topk(k=k, dim=1)
        
        # Create a figure to visualize the retrievals
        fig, axs = plt.subplots(num_examples, k+1, figsize=(3*(k+1), 3*num_examples))
        
        # For each example
        for i in range(num_examples):
            # Show query image
            axs[i, 0].imshow(all_images[i].cpu().permute(1, 2, 0))
            axs[i, 0].set_title("Query Image")
            axs[i, 0].axis('off')
            
            # Show top-k retrievals
            for j in range(k):
                retrieved_idx = topk_indices[i, j].item()
                score = similarity[i, retrieved_idx].item()
                
                # Show retrieved image
                axs[i, j+1].imshow(all_images[retrieved_idx].cpu().permute(1, 2, 0))
                
                # Set title with rank and score
                axs[i, j+1].set_title(f"Rank {j+1}, Score: {score:.2f}")
                
                # Highlight correct match
                if retrieved_idx == i:
                    axs[i, j+1].patch.set_edgecolor('green')
                    axs[i, j+1].patch.set_linewidth(3)
                
                axs[i, j+1].axis('off')
        
        plt.tight_layout()
        
        # Save figure
        save_path = f"{self.save_dir}/{save_prefix}.png"
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        
        return save_path

# Usage example (would be in your main script):
# visualizer = EnhancedVisualization(save_dir='visualizations/')
# visualizer.plot_training_history(history)
# visualizer.visualize_embedding_space(image_embeddings, song_embeddings, method='both', epoch=epoch)
# visualizer.visualize_similarity_matrix(image_embeddings, song_embeddings, temperature=0.05, epoch=epoch)
# metrics, _ = visualizer.plot_recall_analysis(model, test_loader, device)
# visualizer.top_k_retrievals_visualization(model, test_loader, img_folder, device)