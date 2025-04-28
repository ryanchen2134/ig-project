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

def train_contrastive_model(model, train_loader, val_loader, optimizer, loss_fn, 
                        num_epochs=100, patience=15, device='cuda', checkpoint_dir='checkpoints', 
                        visualize_every=5, logger=None, scheduler=None, mixup_alpha=0.2,
                        validate_every=1, eval_recalls=True, temperature_annealing=True):
        """
        Enhanced training function with better evaluation metrics
        """
        # Create checkpoint directory if it doesn't exist
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        if logger is None:
            logger = logging.getLogger('contrastive_training')
        
        model = model.to(device)
        best_val_loss = float('inf')
        best_recall = 0.0  # Track best recall@1
        best_model = None
        no_improve_count = 0
        best_epoch = 0
        
        train_losses = []
        val_losses = []
        recalls = []
        learning_rates = []
        temperatures = []  # If using temperature annealing
        
        # Initial temperature
        temperature = 0.1
        temp_min = 0.01
        
        # Track training time
        start_time = time.time()
        
        logger.info(f"Starting enhanced training on device: {device}")
        
        for epoch in range(num_epochs):
            epoch_start_time = time.time()

            # Training phase
            model.train()
            train_loss = 0.0
            
            # Temperature annealing (gradually decrease temperature)
            if temperature_annealing:
                temperature = max(temp_min, 0.1 * (1.0 - epoch / (num_epochs * 0.8)))
                temperatures.append(temperature)
                if hasattr(loss_fn, 'temperature'):
                    loss_fn.temperature = temperature
            
            for batch_idx, (images, song_embeddings) in enumerate(train_loader):
                images = images.to(device)
                song_embeddings = song_embeddings.to(device)
                
                # Apply mixup augmentation with probability
                if mixup_alpha > 0 and np.random.random() < 0.7:  # 70% probability
                    images, song_embeddings = mixup_batch(images, song_embeddings, alpha=mixup_alpha)
                
                optimizer.zero_grad()
                
                # Forward pass
                image_embeddings, projected_song_embeddings = model(images, song_embeddings)
                
                # Calculate loss
                if hasattr(model, 'similarity'):
                    # Use model's similarity function if available
                    sim_matrix = model.similarity(image_embeddings, projected_song_embeddings)
                    loss = loss_fn(image_embeddings, projected_song_embeddings, use_model_temp=True, model_sim=sim_matrix)
                else:
                    loss = loss_fn(image_embeddings, projected_song_embeddings)
                
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
            
            # Validation phase - only validate every N epochs to speed up training
            if epoch % validate_every == 0:
                model.eval()
                val_loss = 0.0
                
                # For computing recall metrics
                if eval_recalls:
                    all_image_embeddings = []
                    all_song_embeddings = []
                
                with torch.no_grad():
                    for images, song_embeddings in val_loader:
                        images = images.to(device)
                        song_embeddings = song_embeddings.to(device)
                        
                        # Forward pass
                        image_embeddings, projected_song_embeddings = model(images, song_embeddings)
                        
                        # Calculate loss
                        if hasattr(model, 'similarity'):
                            sim_matrix = model.similarity(image_embeddings, projected_song_embeddings)
                            loss = loss_fn(image_embeddings, projected_song_embeddings, use_model_temp=True, model_sim=sim_matrix)
                        else:
                            loss = loss_fn(image_embeddings, projected_song_embeddings)
                        
                        val_loss += loss.item()
                        
                        # Collect embeddings for recall calculation
                        if eval_recalls:
                            all_image_embeddings.append(image_embeddings.cpu())
                            all_song_embeddings.append(projected_song_embeddings.cpu())
                
                avg_val_loss = val_loss / len(val_loader)
                val_losses.append(avg_val_loss)
                
                # Calculate recall metrics
                if eval_recalls:
                    all_image_embeddings = torch.cat(all_image_embeddings, dim=0)
                    all_song_embeddings = torch.cat(all_song_embeddings, dim=0)
                    
                    # Calculate recalls
                    recall_metrics = calculate_recall_metrics(all_image_embeddings, all_song_embeddings)
                    recalls.append(recall_metrics)
                    
                    current_recall = recall_metrics['r@1_i2s']  # Use image->song Recall@1
                    logger.info(f"Recalls: R@1: {recall_metrics['r@1_i2s']:.2f}%, "
                            f"R@5: {recall_metrics['r@5_i2s']:.2f}%, "
                            f"R@10: {recall_metrics['r@10_i2s']:.2f}%")
            else:
                # Fill with previous values when not validating
                if val_losses:
                    val_losses.append(val_losses[-1])
                else:
                    val_losses.append(float('inf'))
                current_recall = best_recall
            
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

            # Calculate epoch time
            epoch_time = time.time() - epoch_start_time
            
            logger.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, "
                    f"Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s")
            
            # Create Visualizations Periodically
            if (epoch + 1) % visualize_every == 0 or epoch == 0 or epoch == num_epochs - 1:
                logger.info("Creating embedding visualizations...")
                visualize_embeddings(
                    all_image_embeddings, all_song_embeddings,
                    method='both', epoch=epoch+1,
                    save_dir=os.path.join(checkpoint_dir, 'visualizations')
                )

            # Early stopping check - use recall@1 instead of loss if evaluating recalls
            improved = False
            if eval_recalls and current_recall > best_recall:
                best_recall = current_recall
                best_model = model.state_dict().copy()
                best_epoch = epoch + 1
                no_improve_count = 0
                improved = True
                logger.info(f"Recall@1 improved to {best_recall:.2f}%")
            elif not eval_recalls and avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model = model.state_dict().copy()
                best_epoch = epoch + 1
                no_improve_count = 0
                improved = True
                logger.info(f"Validation loss improved to {avg_val_loss:.4f}")
                
            if not improved:
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
            'best_val_loss': best_val_loss if not eval_recalls else None,
            'best_recall': best_recall if eval_recalls else None
        }, best_model_path)
        
        # Print training summary
        total_time = time.time() - start_time
        logger.info(f"\nTraining Summary:")
        logger.info(f"Total training time: {total_time:.2f} seconds")
        
        if eval_recalls:
            logger.info(f"Best Recall@1: {best_recall:.2f}% (achieved on epoch {best_epoch})")
        else:
            logger.info(f"Best validation loss: {best_val_loss:.4f} (achieved on epoch {best_epoch})")
        
        history = {
            'train_loss': train_losses,
            'val_loss': val_losses,
            'best_epoch': best_epoch,
            'total_time': total_time,
            'learning_rates': learning_rates,
        }
        
        if eval_recalls:
            history['recalls'] = recalls
            history['best_recall'] = best_recall
        else:
            history['best_val_loss'] = best_val_loss
            
        if temperature_annealing:
            history['temperatures'] = temperatures
        
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

if __name__ == "__main__":
    # Parse command line arguments -- DEFAULT ARGUMENTS
    parser = argparse.ArgumentParser(description='Train contrastive model for image-song matching')
    parser.add_argument('--backbone', type=str, default='resnet18', 
                        choices=['resnet18', 'efficientnet_b0', 'convnext_tiny'],
                        help='Backbone architecture for image encoder')
    parser.add_argument('--batch_size', type=int, default=64, 
                        help='Batch size for training')
    parser.add_argument('--embedding_dim', type=int, default=128, 
                        help='Dimension of embedding space')
    parser.add_argument('--lr', type=float, default=3e-4, 
                        help='Learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4, 
                        help='Weight decay for optimizer')
    parser.add_argument('--epochs', type=int, default=300, 
                        help='Maximum number of epochs')
    parser.add_argument('--patience', type=int, default=15, 
                        help='Early stopping patience')
    parser.add_argument('--temperature', type=float, default=0.05, 
                        help='Temperature parameter for NT-Xent loss')
    parser.add_argument('--hard_negative_weight', type=float, default=0.2, 
                        help='Weight for hard negative mining (0 to disable)')
    parser.add_argument('--mixup_alpha', type=float, default=0.4, 
                        help='Alpha parameter for mixup augmentation (0 to disable)')
    parser.add_argument('--data_path', type=str, default="data/csv files/rawr_dinosaur.csv",
                        help='Path to the CSV data file')
    parser.add_argument('--img_folder', type=str, default="data/final-sample-dataset/images",
                        help='Path to the folder containing images')
    parser.add_argument('--warmup_epochs', type=int, default=10,
                        help='Number of epochs for learning rate warmup')
    parser.add_argument('--min_lr', type=float, default=5e-6,
                        help='Minimum learning rate after cosine decay')
    
    args = parser.parse_args()
    
    # Set up logging first
    logger = setup_logger()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    logger.info("Random seeds set for reproducibility")

    # Parameters from command line
    batch_size = args.batch_size
    embedding_dim = args.embedding_dim
    learning_rate = args.lr
    weight_decay = args.weight_decay
    num_epochs = args.epochs
    patience = args.patience
    temperature = args.temperature
    backbone_type = args.backbone
    hard_negative_weight = args.hard_negative_weight
    mixup_alpha = args.mixup_alpha
    warmup_epochs = args.warmup_epochs
    min_lr = args.min_lr
    
    # Log parameters
    logger.info("Training parameters:")
    logger.info(f"Backbone architecture: {backbone_type}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Weight decay: {weight_decay}")
    logger.info(f"Max epochs: {num_epochs}")
    logger.info(f"Early stopping patience: {patience}")
    logger.info(f"NT-Xent temperature: {temperature}")
    logger.info(f"Hard negative weight: {hard_negative_weight}")
    logger.info(f"Mixup alpha: {mixup_alpha}")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Define your data paths
    data_path = args.data_path
    img_folder = args.img_folder
    
    logger.info(f"Data path: {data_path}")
    logger.info(f"Image folder: {img_folder}")
    
    # Create dataset
    try:
        dataset = ImgSongDataset(file_path=data_path, img_folder=img_folder)
        logger.info(f"Dataset successfully loaded with {len(dataset)} samples")
    except Exception as e:
        logger.error(f"Failed to load dataset: {str(e)}")
        raise
    
    # Get song embedding dimension from the first item
    try:
        _, first_song_features = dataset[0]
        song_embedding_dim = first_song_features.shape[0]  
        logger.info(f"Song embedding dimension: {song_embedding_dim}")
    except Exception as e:
        logger.error(f"Failed to get song embedding dimension: {str(e)}")
        raise
    
    # Split dataset with larger validation set for small datasets
    try:
        train_data, temp_data = train_test_split(dataset, test_size=0.3, random_state=42)
        val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        logger.info(f"Train set: {len(train_data)} samples")
        logger.info(f"Validation set: {len(val_data)} samples")
        logger.info(f"Test set: {len(test_data)} samples")
    except Exception as e:
        logger.error(f"Failed to split dataset: {str(e)}")
        raise
    
    # Create data loaders - adjust batch size if dataset is small
    train_batch_size = min(batch_size, len(train_data))
    val_batch_size = min(batch_size, len(val_data))
    test_batch_size = min(batch_size, len(test_data))
    
    logger.info(f"Actual batch sizes - Train: {train_batch_size}, Val: {val_batch_size}, Test: {test_batch_size}")
    
    try:
        train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True, num_workers=4)
        val_loader = DataLoader(val_data, batch_size=val_batch_size, shuffle=False, num_workers=4)
        test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=False, num_workers=4)
        logger.info("DataLoaders created successfully")
    except Exception as e:
        logger.error(f"Failed to create DataLoaders: {str(e)}")
        raise
    
    # Initialize model
    try:
        model = ContrastiveImageSongModel(
            song_embedding_dim=song_embedding_dim, 
            embedding_dim=embedding_dim,
            backbone_type=backbone_type
        )
        logger.info(f"Model initialized successfully with {backbone_type} backbone")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise
    
    # Loss function and optimizer
    loss_fn = NTXentLoss(temperature=temperature, hard_negative_weight=hard_negative_weight)
    #optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    optimizer = optim.AdamW(model.parameters(), lr = args.lr, weight_decay=args.weight_decay)
    # Learning rate scheduler - cosine annealing with warm restarts works well for small datasets
    scheduler = WarmupCosineScheduler(
        optimizer, 
        warmup_epochs=warmup_epochs,
        max_epochs=num_epochs,
        warmup_start_lr=learning_rate/100,
        base_lr=learning_rate,
        min_lr=min_lr
    )

    logger.info("Optimizer and scheduler initialized")
    logger.info("Starting training...")

    try:
        # Train the model
        trained_model, history = train_contrastive_model(
            model, train_loader, val_loader, optimizer, loss_fn, 
            num_epochs=num_epochs, patience=patience, device=device,
            visualize_every=5, logger=logger, scheduler=scheduler,
            mixup_alpha=mixup_alpha
        )
        
        # Plot training history
        plot_training_history(history, save_path='contrastive_training_history.png', logger=logger)
        
        # Save the model with backbone information
        checkpoint = {
            'model_state_dict': trained_model.state_dict(),
            'embedding_dim': trained_model.embedding_dim,
            'backbone_type': backbone_type,
            'song_embedding_dim': song_embedding_dim
        }
        torch.save(checkpoint, f'contrastive_model_{backbone_type}.pth')
        logger.info(f"Training completed and model saved as contrastive_model_{backbone_type}.pth!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.exception("Stack trace:")
        raise

    # Optional: Evaluate on test set
    logger.info("\nEvaluating on test set...")
    try:
        trained_model.eval()
        test_loss = 0.0
        total_correct_top5 = 0
        total_samples = 0

        with torch.no_grad():
            for images, song_embeddings in test_loader:
                batch_size = images.size(0)  # Important: get actual batch size here
                images = images.to(device)
                song_embeddings = song_embeddings.to(device)

                # Forward pass
                image_embeddings, projected_song_embeddings = trained_model(images, song_embeddings)

                # Compute loss
                loss = loss_fn(image_embeddings, projected_song_embeddings)
                test_loss += loss.item()

                # Compute similarity matrix for the batch
                similarity = torch.matmul(image_embeddings, projected_song_embeddings.T)

                # Top-5 accuracy computation
                _, top5_indices = similarity.topk(k=5, dim=1)
                targets = torch.arange(batch_size).to(device)  # Correct: batch_size here, not global batch_size
                correct_top5 = (top5_indices == targets.view(-1, 1)).any(dim=1).sum().item()

                total_correct_top5 += correct_top5
                total_samples += batch_size  # Add actual batch size

        # Calculate average test loss
        avg_test_loss = test_loss / len(test_loader)
        logger.info(f"Test Loss: {avg_test_loss:.4f}")

        # Calculate top-5 accuracy
        accuracy_top5 = total_correct_top5 / total_samples * 100
        logger.info(f"Top-5 matching accuracy on test set: {accuracy_top5:.2f}%")

    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        logger.exception("Stack trace:")
