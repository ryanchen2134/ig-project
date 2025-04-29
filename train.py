# train.py - Modified for smaller datasets
import torch
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import time
import math
import logging
from graph import visualize_embeddings
from visualizer import EnhancedVisualization
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts, _LRScheduler
import argparse

# Import your modified model and dataset
from models.decoder.contrastive import ContrastiveImageSongModel, NTXentLoss
from data.dloader import ImgSongDataset

# Set up logging
def setup_logger(log_dir='logs'):
    """Set up logger with appropriate configuration"""
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('contrastive_training')
    logger.setLevel(logging.INFO)
    
    # Clear any existing handlers
    if logger.handlers:
        logger.handlers = []
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # Create file handler for logging to file
    log_file = os.path.join(log_dir, f'training_{time.strftime("%Y%m%d_%H%M%S")}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Create console handler for logging to console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    # Add handlers to logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def calculate_recall_metrics(image_embeddings, song_embeddings, ks=[1, 5, 10]):
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
        
        # Create target indices (diagonal)
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

def train_contrastive_model(
    model, train_loader, val_loader, test_loader, optimizer, loss_fn, 
    num_epochs=100, patience=20, device='cuda', 
    save_dir='checkpoints', logger=None, scheduler=None,
    validate_every=1, mixup_alpha=0.4, multiple_views=True,
    visualizer=None):
    """
    Enhanced training function with support for multiple views and improved metrics.
    """
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    if logger is None:
        logger = logging.getLogger('contrastive_training')
    
    # Move model to device
    model = model.to(device)
    
    # Tracking variables
    best_val_loss = float('inf')
    best_recall = 0.0
    best_model = None
    no_improve_count = 0
    best_epoch = 0
    
    # History for plotting
    history = {
        'train_loss': [],
        'val_loss': [],
        'recalls': [],
        'learning_rates': []
    }
    
    logger.info(f"Starting enhanced training with multiple views on device: {device}")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, batch_data in enumerate(train_loader):
            # Handle multiple views
            if multiple_views and isinstance(batch_data[0], list):
                views, song_embeddings = batch_data
                song_embeddings = song_embeddings.to(device)
                
                # Process each view
                all_losses = []
                for view_idx, view in enumerate(views):
                    view = view.to(device)
                    
                    # Apply mixup if enabled
                    if mixup_alpha > 0 and np.random.random() < 0.7:
                        # Get random indices for mixing
                        indices = torch.randperm(view.size(0)).to(device)
                        # Generate mixing coefficient
                        lam = np.random.beta(mixup_alpha, mixup_alpha)
                        # Mix images and song embeddings
                        mixed_view = lam * view + (1 - lam) * view[indices]
                        mixed_song_embeddings = lam * song_embeddings + (1 - lam) * song_embeddings[indices]
                        
                        # Forward pass with mixed data
                        image_embeddings, projected_song_embeddings = model(mixed_view, mixed_song_embeddings)
                    else:
                        # Regular forward pass
                        image_embeddings, projected_song_embeddings = model(view, song_embeddings)
                    
                    # Compute loss
                    loss = loss_fn(image_embeddings, projected_song_embeddings)
                    all_losses.append(loss)
                
                # Average loss across views
                loss = torch.stack(all_losses).mean()
            else:
                # Single view processing
                images, song_embeddings = batch_data
                images = images.to(device)
                song_embeddings = song_embeddings.to(device)
                
                # Apply mixup if enabled
                if mixup_alpha > 0 and np.random.random() < 0.7:
                    indices = torch.randperm(images.size(0)).to(device)
                    lam = np.random.beta(mixup_alpha, mixup_alpha)
                    mixed_images = lam * images + (1 - lam) * images[indices]
                    mixed_song_embeddings = lam * song_embeddings + (1 - lam) * song_embeddings[indices]
                    
                    image_embeddings, projected_song_embeddings = model(mixed_images, mixed_song_embeddings)
                else:
                    image_embeddings, projected_song_embeddings = model(images, song_embeddings)
                
                # Compute loss
                loss = loss_fn(image_embeddings, projected_song_embeddings)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            train_loss += loss.item()
        
        # Calculate average training loss
        avg_train_loss = train_loss / len(train_loader)
        history['train_loss'].append(avg_train_loss)
        
        # Validation phase
        if epoch % validate_every == 0:
            model.eval()
            val_loss = 0.0
            
            # For computing recall metrics
            all_image_embeddings = []
            all_song_embeddings = []
            
            with torch.no_grad():
                for images, song_embeddings in val_loader:
                    images = images.to(device)
                    song_embeddings = song_embeddings.to(device)
                    
                    # Forward pass
                    image_embeddings, projected_song_embeddings = model(images, song_embeddings)
                    
                    # Calculate loss
                    loss = loss_fn(image_embeddings, projected_song_embeddings)
                    val_loss += loss.item()
                    
                    # Collect embeddings for recall calculation
                    all_image_embeddings.append(image_embeddings.cpu())
                    all_song_embeddings.append(projected_song_embeddings.cpu())
            
            # Calculate average validation loss
            avg_val_loss = val_loss / len(val_loader)
            history['val_loss'].append(avg_val_loss)
            
            # Concatenate all embeddings
            all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
            all_song_embeddings = torch.cat(all_song_embeddings, dim=0)
            
            # Calculate recall metrics
            recall_metrics = calculate_recall_metrics(all_image_embeddings, all_song_embeddings)
            history['recalls'].append(recall_metrics)
            
            # Log recalls
            logger.info(f"Recalls: R@1: {recall_metrics['r@1_i2s']:.2f}%, "
                      f"R@5: {recall_metrics['r@5_i2s']:.2f}%, "
                      f"R@10: {recall_metrics['r@10_i2s']:.2f}%")
            
            # Visualize embeddings if visualizer provided
            if visualizer is not None and (epoch % 5 == 0 or epoch == num_epochs - 1):
                visualizer.visualize_embedding_space(
                    all_image_embeddings, all_song_embeddings,
                    method='both', epoch=epoch+1
                )
                visualizer.visualize_similarity_matrix(
                    all_image_embeddings, all_song_embeddings,
                    temperature=loss_fn.temperature, epoch=epoch+1
                )
            
            # Step scheduler if provided
            if scheduler is not None:
                if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                    scheduler.step(avg_val_loss)
                else:
                    scheduler.step()
                
                # Log current learning rate
                current_lr = optimizer.param_groups[0]['lr']
                history['learning_rates'].append(current_lr)
                logger.info(f"Current Learning rate: {current_lr:.6f}")
            
            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
                      f"Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s")
            
            # Check for improvement - use R@1 for early stopping
            current_recall = recall_metrics['r@1_i2s']
            
            if current_recall > best_recall:
                best_recall = current_recall
                best_model = model.state_dict().copy()
                best_epoch = epoch + 1
                no_improve_count = 0
                logger.info(f"Recall@1 improved to {best_recall:.2f}%")
                
                # Save best model
                best_model_path = os.path.join(save_dir, 'best_model.pth')
                torch.save({
                    'epoch': best_epoch,
                    'model_state_dict': model.state_dict(),
                    'embedding_dim': model.embedding_dim,
                    'backbone_type': model.backbone_type,
                    'best_recall': best_recall
                }, best_model_path)
            else:
                no_improve_count += 1
                logger.info(f"No improvement for {no_improve_count} epochs")
                
                if no_improve_count >= patience:
                    logger.info(f"Early stopping triggered after {epoch+1} epochs")
                    break
    
    # Training completed, evaluate on test set
    if best_model is not None:
        model.load_state_dict(best_model)
    
    # Evaluate on test set
    logger.info("\nEvaluating on test set...")
    model.eval()
    test_loss = 0.0
    all_image_embeddings = []
    all_song_embeddings = []
    
    with torch.no_grad():
        for images, song_embeddings in test_loader:
            images = images.to(device)
            song_embeddings = song_embeddings.to(device)
            
            # Forward pass
            image_embeddings, projected_song_embeddings = model(images, song_embeddings)
            
            # Compute loss
            loss = loss_fn(image_embeddings, projected_song_embeddings)
            test_loss += loss.item()
            
            # Collect embeddings
            all_image_embeddings.append(image_embeddings.cpu())
            all_song_embeddings.append(projected_song_embeddings.cpu())
    
    # Calculate test metrics
    avg_test_loss = test_loss / len(test_loader)
    logger.info(f"Test Loss: {avg_test_loss:.4f}")
    
    # Concatenate all embeddings
    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
    all_song_embeddings = torch.cat(all_song_embeddings, dim=0)
    
    # Calculate recall metrics
    recall_metrics = calculate_recall_metrics(all_image_embeddings, all_song_embeddings, ks=[1, 5, 10, 50])
    
    # Log test recalls
    logger.info(f"Test Recalls: R@1: {recall_metrics['r@1_i2s']:.2f}%, "
              f"R@5: {recall_metrics['r@5_i2s']:.2f}%, "
              f"R@10: {recall_metrics['r@10_i2s']:.2f}%, "
              f"R@50: {recall_metrics['r@50_i2s']:.2f}%")
    
    # If visualizer is provided, create final visualizations
    if visualizer is not None:
        # Embedding space visualization
        visualizer.visualize_embedding_space(
            all_image_embeddings, all_song_embeddings,
            method='both', epoch='final'
        )
        
        # Similarity matrix
        visualizer.visualize_similarity_matrix(
            all_image_embeddings, all_song_embeddings,
            temperature=loss_fn.temperature, epoch='final'
        )
        
        # Detailed recall analysis
        visualizer.plot_recall_analysis(model, test_loader, device)
    
    # Calculate total training time
    total_time = time.time() - start_time
    
    # Update history
    history['best_epoch'] = best_epoch
    history['best_recall'] = best_recall
    history['test_metrics'] = recall_metrics
    history['total_time'] = total_time
    
    # Print training summary
    logger.info(f"\nTraining Summary:")
    logger.info(f"Total training time: {total_time:.2f} seconds")
    logger.info(f"Best Recall@1: {best_recall:.2f}% (achieved on epoch {best_epoch})")
    
    return model, history

class WarmupCosineScheduler(_LRScheduler):
    """
    Implements a learning rate scheduler with warmup followed by cosine annealing.
    
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
                 base_lr=5e-4, min_lr=1e-5, last_epoch=-1):
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

# Mixup augmentation for contrastive learning - helps small datasets
def mixup_batch(images, song_embeddings, alpha=0.2):
    """
    Applies mixup augmentation to the batch
    
    Args:
        images: Batch of images (B, C, H, W)
        song_embeddings: Batch of song embeddings (B, D)
        alpha: Mixup interpolation strength
        
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

def plot_training_history(history, save_path=None, logger=None):
    """Plot training and validation loss"""
    if logger is None:
        logger = logging.getLogger('contrastive_training')
        
    logger.info("Plotting training history...")
    
    plt.figure(figsize=(12, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.axvline(x=history.get('best_epoch', 0) - 1, color='r', linestyle='--', label='Best Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Contrastive Learning Training History')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot loss on log scale to better see small changes
    plt.subplot(2, 1, 2)
    plt.semilogy(history['train_loss'], label='Training Loss')
    plt.semilogy(history['val_loss'], label='Validation Loss')
    plt.axvline(x=history.get('best_epoch', 0) - 1, color='r', linestyle='--', label='Best Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss (Logarithmic Scale)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = 'contrastive_training_history.png'
    
    plt.savefig(save_path)
    logger.info(f"Training history plot saved to {save_path}")
    plt.show()

def main(args):
    """Main training pipeline with all improvements"""
    # Set up logging
    logger = setup_logger()
    
    # Set random seeds for reproducibility
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    logger.info("Random seeds set for reproducibility")
    
    # Log training parameters
    logger.info("Training parameters:")
    logger.info(f"Backbone architecture: {args.backbone}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Embedding dimension: {args.embedding_dim}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Weight decay: {args.weight_decay}")
    logger.info(f"Dropout: {args.dropout}")
    logger.info(f"Max epochs: {args.epochs}")
    logger.info(f"Early stopping patience: {args.patience}")
    logger.info(f"NT-Xent temperature: {args.temperature}")
    logger.info(f"Hard negative weight: {args.hard_negative_weight}")
    logger.info(f"Mixup alpha: {args.mixup_alpha}")
    logger.info(f"Number of views: {args.num_views}")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create datasets with multiple views for training
    train_dataset = ImgSongDataset(
        file_path=args.data_path,
        img_folder=args.img_folder,
        num_views=args.num_views,
        is_train=True
    )
    logger.info(f"Training dataset loaded with {len(train_dataset)} samples")
    
    # Validation and test datasets (single view)
    val_dataset = ImgSongDataset(
        file_path=args.data_path,
        img_folder=args.img_folder,
        is_train=False
    )
    test_dataset = ImgSongDataset(
        file_path=args.data_path,
        img_folder=args.img_folder,
        is_train=False
    )
    
    # Split datasets
    train_indices, temp_indices = train_test_split(
        np.arange(len(train_dataset)), 
        test_size=0.3, 
        random_state=args.seed
    )
    val_indices, test_indices = train_test_split(
        temp_indices, 
        test_size=0.5, 
        random_state=args.seed
    )
    
    # Create subset datasets
    from torch.utils.data import Subset
    train_data = Subset(train_dataset, train_indices)
    val_data = Subset(val_dataset, val_indices)
    test_data = Subset(test_dataset, test_indices)
    
    logger.info(f"Dataset split: Train={len(train_data)}, Val={len(val_data)}, Test={len(test_data)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_data, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.num_workers
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    test_loader = DataLoader(
        test_data, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.num_workers
    )
    
    # Get song embedding dimension
    _, first_song_features = train_dataset[0]
    if isinstance(first_song_features, list):  # Multiple views case
        song_embedding_dim = first_song_features[0].shape[0]
    else:
        song_embedding_dim = first_song_features.shape[0]
    
    logger.info(f"Song embedding dimension: {song_embedding_dim}")
    
    # Initialize improved model
    model = ContrastiveImageSongModel(
        song_embedding_dim=song_embedding_dim,
        embedding_dim=args.embedding_dim,
        backbone_type=args.backbone,
        dropout=args.dropout
    )
    logger.info(f"Model initialized with {args.backbone} backbone")
    
    # Enhanced loss function
    loss_fn = NTXentLoss(
        temperature=args.temperature,
        hard_negative_weight=args.hard_negative_weight,
        use_hard_negatives=args.hard_negative_weight > 0
    )
    
    # Optimizer (AdamW instead of Adam)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    print(f"Initial learning rate: {optimizer.param_groups[0]['lr']}")  # Add this debug line
    
    # Learning rate scheduler with warmup
    from transformers import get_cosine_schedule_with_warmup
    
    # Calculate number of training steps
    num_training_steps = args.epochs * len(train_loader)
    warmup_steps = int(num_training_steps * 0.1)  # 10% warmup
    
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=num_training_steps
    )
    
    logger.info("Optimizer and scheduler initialized")
    
    # Initialize visualizer
    visualizer = EnhancedVisualization(save_dir=os.path.join(args.save_dir, 'visualizations'))
    
    # Train the model with all enhancements
    logger.info("Starting enhanced training...")
    trained_model, history = train_contrastive_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=args.epochs,
        patience=args.patience,
        device=device,
        save_dir=args.save_dir,
        logger=logger,
        scheduler=scheduler,
        mixup_alpha=args.mixup_alpha,
        multiple_views=args.num_views > 1,
        visualizer=visualizer
    )
    
    # Plot enhanced training history
    visualizer.plot_training_history(history)
    
    # Save final model
    final_model_path = os.path.join(args.save_dir, f'final_model_{args.backbone}.pth')
    torch.save({
        'model_state_dict': trained_model.state_dict(),
        'embedding_dim': trained_model.embedding_dim,
        'backbone_type': trained_model.backbone_type,
        'song_embedding_dim': song_embedding_dim,
        'history': history
    }, final_model_path)
    
    logger.info(f"Training completed and model saved to {final_model_path}")
    
    return trained_model, history

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Contrastive Learning for Image-Song Matching")
    
    # Model parameters
    parser.add_argument('--backbone', type=str, default='efficientnet_b3', 
                      choices=['efficientnet_b0', 'efficientnet_b3', 'resnet18', 'resnet50'],
                      help='Backbone architecture for image encoder')
    parser.add_argument('--embedding_dim', type=int, default=256, 
                      help='Dimension of embedding space')
    parser.add_argument('--dropout', type=float, default=0.3,
                      help='Dropout rate for projection head')
    
    # Training parameters
    parser.add_argument('--batch_size', type=int, default=32, 
                      help='Batch size for training')
    parser.add_argument('--lr', type=float, default=1e-4, 
                      help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.05, 
                      help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, default=100, 
                      help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=20, 
                      help='Early stopping patience')
    parser.add_argument('--num_workers', type=int, default=4,
                      help='Number of workers for data loading')
    parser.add_argument('--seed', type=int, default=42,
                      help='Random seed for reproducibility')
    
    # Loss function parameters
    parser.add_argument('--temperature', type=float, default=0.05, 
                      help='Temperature parameter for NT-Xent loss')
    parser.add_argument('--hard_negative_weight', type=float, default=0.5, 
                      help='Weight for hard negative mining (0 to disable)')
    
    # Data augmentation parameters
    parser.add_argument('--mixup_alpha', type=float, default=0.4, 
                      help='Alpha parameter for mixup augmentation (0 to disable)')
    parser.add_argument('--num_views', type=int, default=2,
                      help='Number of views per image for contrastive learning')
    
    # Paths
    parser.add_argument('--data_path', type=str, default="data/csv files/rawr_dinosaur.csv",
                      help='Path to the CSV data file')
    parser.add_argument('--img_folder', type=str, default="data/final-sample-dataset/images",
                      help='Path to the folder containing images')
    parser.add_argument('--save_dir', type=str, default="checkpoints",
                      help='Directory to save checkpoints and visualizations')
    
    args = parser.parse_args()
    
    # Run training pipeline
    model, history = main(args)
