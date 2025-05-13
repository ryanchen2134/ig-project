import torch
import numpy as np
from typing import Dict, List, Tuple, Optional

def calculate_recall_at_k(image_embeddings: torch.Tensor, 
                         song_embeddings: torch.Tensor, 
                         ks: List[int] = [1, 5, 10]) -> Dict[str, float]:
    """
    Calculate recall metrics between image and song embeddings
    
    Args:
        image_embeddings: Image embeddings tensor
        song_embeddings: Song embeddings tensor
        ks: List of k values for Recall@k
        
    Returns:
        Dictionary of recall metrics
    """
    # Calculate similarity matrix
    similarity = torch.matmul(image_embeddings, song_embeddings.T)
    
    # Get the size
    batch_size = image_embeddings.size(0)
    
    # Calculate recalls for image->song direction
    metrics = {}
    for k in ks:
        # Get top-k indices
        _, indices_i2s = similarity.topk(k=min(k, batch_size), dim=1)
        
        # Create target indices (diagonal - matching pairs)
        targets = torch.arange(batch_size).view(-1, 1)
        
        # Check if target index is in top-k
        correct_i2s = torch.any(indices_i2s == targets, dim=1).float().mean().item() * 100
        metrics[f'r@{k}_i2s'] = correct_i2s
        
        # Repeat for song->image direction
        _, indices_s2i = similarity.T.topk(k=min(k, batch_size), dim=1)
        correct_s2i = torch.any(indices_s2i == targets, dim=1).float().mean().item() * 100
        metrics[f'r@{k}_s2i'] = correct_s2i
        
        # Overall recall (average of both directions)
        metrics[f'r@{k}'] = (correct_i2s + correct_s2i) / 2
    
    return metrics

def mean_average_precision(image_embeddings: torch.Tensor, 
                          song_embeddings: torch.Tensor) -> float:
    """
    Calculate mean average precision (mAP) for retrieval
    
    Args:
        image_embeddings: Image embeddings
        song_embeddings: Song embeddings
        
    Returns:
        Mean average precision score
    """
    batch_size = image_embeddings.size(0)
    
    # Calculate similarity scores
    similarity = torch.matmul(image_embeddings, song_embeddings.T)
    
    # Get ranking of correct match for each query
    _, indices = similarity.sort(descending=True, dim=1)
    
    # Find position of correct match (ground truth)
    # For each image, its matching song has the same index
    ground_truth = torch.arange(batch_size, device=image_embeddings.device)
    
    # Create a mask of where the correct matches are in the sorted results
    matches = (indices == ground_truth.view(-1, 1))
    
    # Find the positions (ranks) where matches occur
    ranks = torch.nonzero(matches)[:, 1].float() + 1
    
    # Calculate average precision: AP = 1/rank
    ap_scores = 1.0 / ranks
    
    # Return mean of average precision scores
    return ap_scores.mean().item()

def similarity_distribution(image_embeddings: torch.Tensor, 
                          song_embeddings: torch.Tensor) -> Dict[str, float]:
    """
    Calculate statistics about similarity distribution
    
    Args:
        image_embeddings: Image embeddings
        song_embeddings: Song embeddings
        
    Returns:
        Dictionary with similarity statistics
    """
    # Calculate similarity matrix
    similarity = torch.matmul(image_embeddings, song_embeddings.T)
    
    # Get diagonal (matching pairs) and off-diagonal (non-matching) elements
    matching_sim = torch.diagonal(similarity)
    
    # Create mask for off-diagonal elements
    mask = ~torch.eye(len(image_embeddings), dtype=bool, device=similarity.device)
    non_matching_sim = similarity[mask]
    
    # Calculate statistics
    stats = {
        'matching_mean': matching_sim.mean().item(),
        'matching_std': matching_sim.std().item(),
        'matching_min': matching_sim.min().item(),
        'matching_max': matching_sim.max().item(),
        'non_matching_mean': non_matching_sim.mean().item(),
        'non_matching_std': non_matching_sim.std().item(),
        'non_matching_min': non_matching_sim.min().item(),
        'non_matching_max': non_matching_sim.max().item(),
        'contrast': matching_sim.mean().item() - non_matching_sim.mean().item()
    }
    
    return stats