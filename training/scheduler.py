import torch
import math
from torch.optim.lr_scheduler import _LRScheduler

class WarmupCosineScheduler(_LRScheduler):
    """
    Learning rate scheduler with warmup followed by cosine annealing.
    
    Args:
        optimizer: The optimizer to adjust learning rate for
        warmup_epochs: Number of epochs for linear warmup
        max_epochs: Total number of epochs
        warmup_start_lr: Initial learning rate for warmup phase
        base_lr: Learning rate after warmup (peak learning rate)
        min_lr: Minimum learning rate after cosine decay
        last_epoch: The index of last epoch
    """
    def __init__(self, optimizer, warmup_epochs, max_epochs, warmup_start_lr=1e-6, 
                 base_lr=1e-4, min_lr=1e-5, last_epoch=-1):
        self.warmup_epochs = warmup_epochs
        self.max_epochs = max_epochs
        self.warmup_start_lr = warmup_start_lr
        self.base_lr = base_lr
        self.min_lr = min_lr
        super(WarmupCosineScheduler, self).__init__(optimizer, last_epoch)
        
    def get_lr(self):
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            alpha = self.last_epoch / self.warmup_epochs
            lr_scale = self.warmup_start_lr + alpha * (self.base_lr - self.warmup_start_lr)
        else:
            # Cosine decay after warmup
            progress = (self.last_epoch - self.warmup_epochs) / (self.max_epochs - self.warmup_epochs)
            progress = min(1.0, progress)  # Ensure we don't go beyond 1.0
            cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
            lr_scale = self.min_lr + cosine_decay * (self.base_lr - self.min_lr)
            
        return [lr_scale for _ in self.base_lrs]