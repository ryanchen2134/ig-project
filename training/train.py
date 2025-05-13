import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
import numpy as np
import os
import time
import logging
from typing import Dict, Tuple, Optional, Any, List

def train_contrastive_model(
    model: torch.nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn: torch.nn.Module,
    num_epochs: int = 100,
    patience: int = 15,
    device: str = 'cuda',
    checkpoint_dir: str = 'checkpoints',
    visualize_every: int = 5,
    logger: Optional[logging.Logger] = None,
    scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None,
    mixup_alpha: float = 0.2
) -> Tuple[torch.nn.Module, Dict[str, Any]]:
    """
    Train a contrastive learning model
    
    Args:
        model: The model to train
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimizer
        loss_fn: Loss function
        num_epochs: Number of epochs to train for
        patience: Early stopping patience
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        visualize_every: How often to visualize embeddings
        logger: Logger
        scheduler: Learning rate scheduler
        mixup_alpha: Alpha parameter for mixup augmentation
        
    Returns:
        Tuple of (trained_model, training_history)
    """
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    if logger is None:
        logger = logging.getLogger('contrastive_training')
    
    model = model.to(device)
    best_val_loss = float('inf')
    best_model = None
    no_improve_count = 0
    best_epoch = 0
    
    train_losses = []
    val_losses = []
    learning_rates = []
    
    start_time = time.time()
    
    logger.info(f"Starting training on device: {device}")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()

        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, song_embeddings) in enumerate(train_loader):
            images = images.to(device)
            song_embeddings = song_embeddings.to(device)
            
            # Apply mixup augmentation
            if mixup_alpha > 0:
                images, song_embeddings = _mixup_batch(images, song_embeddings, alpha=mixup_alpha)
            
            optimizer.zero_grad()
            
            # Forward pass
            image_embeddings, projected_song_embeddings = model(images, song_embeddings)
            
            # Calculate loss using model's similarity function if available
            if hasattr(model, 'similarity'):
                sim_matrix = model.similarity(image_embeddings, projected_song_embeddings)
                loss = loss_fn(image_embeddings, projected_song_embeddings, 
                                use_model_temp=True, model_sim=sim_matrix)
            else:
                loss = loss_fn(image_embeddings, projected_song_embeddings)
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Training loss is {loss.item()}, stopping training")
                return model, {'train_loss': train_losses, 'val_loss': val_losses}
            
            # Backward pass and optimize
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, song_embeddings in val_loader:
                images = images.to(device)
                song_embeddings = song_embeddings.to(device)
                
                # Forward pass
                image_embeddings, projected_song_embeddings = model(images, song_embeddings)
                
                # Calculate loss
                if hasattr(model, 'similarity'):
                    sim_matrix = model.similarity(image_embeddings, projected_song_embeddings)
                    loss = loss_fn(image_embeddings, projected_song_embeddings, 
                                    use_model_temp=True, model_sim=sim_matrix)
                else:
                    loss = loss_fn(image_embeddings, projected_song_embeddings)
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Step the scheduler if provided
        if scheduler is not None:
            if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(avg_val_loss)
            else:
                scheduler.step()
            
            # Log the current learning rate
            current_lr = optimizer.param_groups[0]['lr']
            learning_rates.append(current_lr)
            logger.info(f"Current Learning rate: {current_lr:.6f}")

        # Log epoch results
        epoch_time = time.time() - epoch_start_time
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], "
                    f"Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, "
                    f"Time: {epoch_time:.2f}s")
        
        # Save checkpoint and visualize
        if (epoch + 1) % visualize_every == 0 or epoch == 0 or epoch == num_epochs - 1:
            _save_checkpoint(model, optimizer, epoch, checkpoint_dir, avg_val_loss)
            
            # Visualize embeddings
            from training.visualize import visualize_embeddings
            
            # Collect embeddings for visualization
            all_img_embs, all_song_embs = _collect_embeddings(model, val_loader, device)
            
            visualize_embeddings(
                all_img_embs, all_song_embs,
                method='tsne', epoch=epoch+1,
                save_dir=os.path.join(checkpoint_dir, 'visualizations')
            )

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
            best_epoch = epoch + 1
            no_improve_count = 0
            logger.info(f"Validation loss improved to {avg_val_loss:.4f}")
        else:
            no_improve_count += 1
            logger.info(f"No improvement for {no_improve_count} epochs")
            
            if no_improve_count >= patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(best_model)
    
    # Save best model
    best_model_path = os.path.join(checkpoint_dir, 'best_model.pth')
    torch.save({
        'epoch': best_epoch,
        'model_state_dict': model.state_dict(),
        'embedding_dim': model.embedding_dim,
        'best_val_loss': best_val_loss
    }, best_model_path)
    
    # Print training summary
    total_time = time.time() - start_time
    logger.info(f"\nTraining Summary:")
    logger.info(f"Total training time: {total_time:.2f} seconds")
    logger.info(f"Best validation loss: {best_val_loss:.4f} (achieved on epoch {best_epoch})")
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'best_val_loss': best_val_loss,
        'best_epoch': best_epoch,
        'total_time': total_time,
        'learning_rates': learning_rates
    }
    
    return model, history

def _mixup_batch(images: torch.Tensor, 
                song_embeddings: torch.Tensor, 
                alpha: float = 0.2) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Apply mixup augmentation to a batch
    
    Args:
        images: Image tensor
        song_embeddings: Song embedding tensor
        alpha: Mixup alpha parameter
        
    Returns:
        Mixed images and embeddings
    """
    if alpha <= 0:
        return images, song_embeddings
        
    batch_size = images.size(0)
    
    # Generate random indices for mixing
    indices = torch.randperm(batch_size).to(images.device)
    
    # Generate random mixing coefficient
    lam = np.random.beta(alpha, alpha)
    
    # Mix images
    mixed_images = lam * images + (1 - lam) * images[indices]
    
    # Mix song embeddings
    mixed_song_embeddings = lam * song_embeddings + (1 - lam) * song_embeddings[indices]
    
    return mixed_images, mixed_song_embeddings

def _save_checkpoint(model: torch.nn.Module, 
                    optimizer: torch.optim.Optimizer, 
                    epoch: int, 
                    checkpoint_dir: str, 
                    val_loss: float) -> None:
    """
    Save a checkpoint of the model
    
    Args:
        model: Model to save
        optimizer: Optimizer
        epoch: Current epoch
        checkpoint_dir: Directory to save to
        val_loss: Validation loss
    """
    checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
    torch.save({
        'epoch': epoch + 1,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_loss': val_loss,
    }, checkpoint_path)
    
def _collect_embeddings(model: torch.nn.Module, 
                        data_loader: DataLoader, 
                        device: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collect embeddings from a dataloader
    
    Args:
        model: Model to use
        data_loader: DataLoader for data
        device: Device to run on
        
    Returns:
        Tuple of image and song embeddings
    """
    model.eval()
    img_embeds = []
    song_embeds = []
    
    with torch.no_grad():
        for images, song_features in data_loader:
            images = images.to(device)
            song_features = song_features.to(device)
            
            # Forward pass
            image_embeddings, song_embeddings = model(images, song_features)
            
            # Collect embeddings
            img_embeds.append(image_embeddings.cpu())
            song_embeds.append(song_embeddings.cpu())
    
    return torch.cat(img_embeds, dim=0), torch.cat(song_embeds, dim=0)