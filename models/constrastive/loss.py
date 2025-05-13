import torch
import torch.nn as nn
import torch.nn.functional as F

class NTXentLoss(nn.Module):
    def __init__(self, temperature: float = 0.1, 
                 hard_negative_weight: float = 0.3, 
                 hardest_only: bool = False):
        """        
        NT-Xent loss with hard negative mining
        
        Args:
            temperature: Temperature parameter (not used if model has learned scaling)
            hard_negative_weight: Weight for hard negative mining
            hardest_only: If True, only consider the hardest negative
        """
        super(NTXentLoss, self).__init__()
        self.temperature = temperature
        self.hard_negative_weight = hard_negative_weight
        self.hardest_only = hardest_only
        self.criterion = nn.CrossEntropyLoss()
        self.margin = 0.3  
        
    def forward(self, image_embeddings: torch.Tensor, 
                song_embeddings: torch.Tensor, 
                use_model_temp: bool = True, 
                model_sim: torch.Tensor = None) -> torch.Tensor:
        """
        Compute NT-Xent loss with hard negative mining
        
        Args:
            image_embeddings: Image embeddings
            song_embeddings: Song embeddings
            use_model_temp: Whether to use model's temperature
            model_sim: Pre-computed similarity matrix (optional)
            
        Returns:
            Loss value
        """
        batch_size = image_embeddings.size(0)
        
        # Use provided similarity or compute it
        if model_sim is not None:
            similarity_matrix = model_sim
        else:
            similarity_matrix = torch.matmul(image_embeddings, song_embeddings.T)
            if not use_model_temp:
                similarity_matrix = similarity_matrix / self.temperature
        
        # For InfoNCE loss, the positive samples are the diagonal elements
        labels = torch.arange(batch_size).to(similarity_matrix.device)
        
        # Get positive pair similarities (diagonal)
        pos_sim = torch.diag(similarity_matrix)
        
        if self.hard_negative_weight > 0:
            similarity_matrix_detached = similarity_matrix.detach().clone()
            
            # Create mask for positives
            mask = torch.eye(batch_size, dtype=torch.bool, device=similarity_matrix_detached.device)
            similarity_matrix_detached.masked_fill_(mask, float('-inf'))
            
            # Find hard negatives (highest similarity incorrect matches)
            hard_negatives_values, hard_negatives = torch.topk(similarity_matrix_detached, 
                                                             k=2 if not self.hardest_only else 1, 
                                                             dim=1)
            
            # Create boosted similarity matrix
            boosted_sim = similarity_matrix.clone()
            
            # Apply weighting to the hardest negatives
            for i in range(boosted_sim.shape[0]):
                for j, neg_idx in enumerate(hard_negatives[i]):
                    # Weight decreases as we move from hardest to less hard
                    weight = self.hard_negative_weight / (j + 1) if not self.hardest_only else self.hard_negative_weight
                    boosted_sim[i, neg_idx] *= (1 + weight)
            
            # Standard InfoNCE losses with hard negatives
            loss_i2s = self.criterion(boosted_sim, labels)
            loss_s2i = self.criterion(boosted_sim.T, labels)
            
            # Add margin-based contrastive component for most difficult negatives
            neg_sim = hard_negatives_values[:, 0]  
            margin_loss = torch.mean(torch.clamp(neg_sim - pos_sim + self.margin, min=0))
            
            # Combined loss
            total_loss = (loss_i2s + loss_s2i) / 2 + 0.3 * margin_loss
            
        else:
            # Standard NT-Xent loss 
            loss_i2s = self.criterion(similarity_matrix, labels)
            loss_s2i = self.criterion(similarity_matrix.T, labels)
            total_loss = (loss_i2s + loss_s2i) / 2
        
        return total_loss