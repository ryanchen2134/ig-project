# train.py
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import os
import time
import logging
from graph import visualize_embeddings

# Import your model and dataset
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

def train_contrastive_model(model, train_loader, val_loader, optimizer, loss_fn, 
                           num_epochs=50, patience=10, device='cuda', checkpoint_dir='checkpoints', 
                           visualize_every=10, logger=None):
    """
    Train the contrastive model
    
    Args:
        model: The contrastive model
        train_loader: DataLoader for training data
        val_loader: DataLoader for validation data
        optimizer: Optimization algorithm
        loss_fn: Contrastive loss function
        num_epochs: Maximum number of epochs
        patience: Early stopping patience
        device: Device to train on
        checkpoint_dir: Directory to save checkpoints
        visualize_every: How often to visualize embeddings
        logger: Logger instance
    
    Returns:
        model: Trained model
        history: Training history
    """
    # Create checkpoint directory if it doesn't exist
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Use default logger if not provided
    if logger is None:
        logger = logging.getLogger('contrastive_training')
    
    model = model.to(device)
    best_val_loss = float('inf')
    best_model = None
    no_improve_count = 0
    best_epoch = 0
    
    train_losses = []
    val_losses = []
    
    # Track training time
    start_time = time.time()
    
    logger.info(f"Starting training on device: {device}")
    logger.info(f"Number of training batches: {len(train_loader)}")
    logger.info(f"Number of validation batches: {len(val_loader)}")
    
    for epoch in range(num_epochs):
        epoch_start_time = time.time()
        
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, song_embeddings) in enumerate(train_loader):
            images = images.to(device)
            song_embeddings = song_embeddings.to(device)
            
            optimizer.zero_grad()
            
            # Forward pass
            image_embeddings, projected_song_embeddings = model(images, song_embeddings)
            
            # Calculate loss
            loss = loss_fn(image_embeddings, projected_song_embeddings)
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                logger.error(f"Training loss is {loss.item()}, stopping training")
                return model, {'train_loss': train_losses, 'val_loss': val_losses}
            
            # Backward pass and optimize
            loss.backward()
            
            # Add gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                logger.debug(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
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
                loss = loss_fn(image_embeddings, projected_song_embeddings)
                
                if torch.isnan(loss) or torch.isinf(loss):
                    logger.error(f"Validation loss is {loss.item()}, stopping training")
                    return model, {'train_loss': train_losses, 'val_loss': val_losses}
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Calculate epoch time
        epoch_time = time.time() - epoch_start_time
        
        logger.info(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Time: {epoch_time:.2f}s")
        
        # Save checkpoint every 10 epochs
        if (epoch + 1) % 10 == 0:
            checkpoint_path = os.path.join(checkpoint_dir, f'checkpoint_epoch_{epoch+1}.pth')
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_losses,
                'val_loss': val_losses,
            }, checkpoint_path)
            logger.info(f"Checkpoint saved to {checkpoint_path}")
        
        # Create Visualizations Periodically
        if (epoch + 1) % visualize_every == 0 or epoch == 0 or epoch == num_epochs - 1:
            logger.info("Creating embedding visualizations...")
            # Collect a subset of embeddings for visualization
            model.eval()
            max_samples = 200  # Limit samples to avoid cluttered plots
            
            image_embeds_list = []
            song_embeds_list = []
            
            with torch.no_grad():
                # Get a subset of validation data
                sample_count = 0
                for images, song_embeddings in val_loader:
                    if sample_count >= max_samples:
                        break
                        
                    batch_size = images.shape[0]
                    if sample_count + batch_size > max_samples:
                        # Take only what we need to reach max_samples
                        images = images[:max_samples - sample_count]
                        song_embeddings = song_embeddings[:max_samples - sample_count]
                    
                    images = images.to(device)
                    song_embeddings = song_embeddings.to(device)
                    
                    # Get embeddings
                    image_embeddings, projected_song_embeddings = model(images, song_embeddings)
                    
                    # Store embeddings
                    image_embeds_list.append(image_embeddings.cpu())
                    song_embeds_list.append(projected_song_embeddings.cpu())
                    
                    sample_count += images.shape[0]
            
            if image_embeds_list:
                # Concatenate all collected embeddings
                all_image_embeddings = torch.cat(image_embeds_list, dim=0)
                all_song_embeddings = torch.cat(song_embeds_list, dim=0)
                
                # Create visualizations with both methods
                visualize_embeddings(
                    all_image_embeddings, all_song_embeddings,
                    method='both', epoch=epoch+1,
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
    logger.info(f"Best model saved to {best_model_path}")
    
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
        'total_time': total_time
    }
    
    return model, history

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
    # Set up logging first
    logger = setup_logger()
    
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    logger.info("Random seeds set for reproducibility")

    # Parameters
    batch_size = 16
    embedding_dim = 64
    learning_rate = 5e-5
    weight_decay = 1e-4
    num_epochs = 100
    patience = 15
    temperature = 0.1  # Temperature parameter for NT-Xent loss
    
    # Log parameters
    logger.info("Training parameters:")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Embedding dimension: {embedding_dim}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Weight decay: {weight_decay}")
    logger.info(f"Max epochs: {num_epochs}")
    logger.info(f"Early stopping patience: {patience}")
    logger.info(f"NT-Xent temperature: {temperature}")
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Define your data paths here
    data_path = "essentia_audio_encoder/final-sample-dataset/data.csv"  # CSV FILE PATH HERE
    img_folder = "essentia_audio_encoder/final-sample-dataset/images"
    
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
    
    # Split dataset
    try:
        train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
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
        model = ContrastiveImageSongModel(song_embedding_dim=song_embedding_dim, embedding_dim=embedding_dim)
        logger.info("Model initialized successfully")
    except Exception as e:
        logger.error(f"Failed to initialize model: {str(e)}")
        raise
    
    # Loss function and optimizer
    loss_fn = NTXentLoss(temperature=temperature, batch_size=train_batch_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )
    logger.info("Optimizer and scheduler initialized")
    
    logger.info("Starting training...")
    
    try:
        # Train the model
        trained_model, history = train_contrastive_model(
            model, train_loader, val_loader, optimizer, loss_fn, 
            num_epochs=num_epochs, patience=patience, device=device,
            visualize_every=10, logger=logger  # Pass the logger
        )
        
        # Plot training history
        plot_training_history(history, save_path='contrastive_training_history.png', logger=logger)
        
        # Save the model
        torch.save(trained_model.state_dict(), 'contrastive_image_song_model.pth')
        logger.info("Training completed and model saved!")
        
    except Exception as e:
        logger.error(f"Error during training: {str(e)}")
        logger.exception("Stack trace:")
        raise
    
    # Optional: Evaluate on test set
    logger.info("\nEvaluating on test set...")
    
    try:
        trained_model.eval()
        test_loss = 0.0
        
        with torch.no_grad():
            for images, song_embeddings in test_loader:
                images = images.to(device)
                song_embeddings = song_embeddings.to(device)
                
                image_embeddings, projected_song_embeddings = trained_model(images, song_embeddings)
                loss = loss_fn(image_embeddings, projected_song_embeddings)
                test_loss += loss.item()
        
        avg_test_loss = test_loss / len(test_loader)
        logger.info(f"Test Loss: {avg_test_loss:.4f}")
    except Exception as e:
        logger.error(f"Error during testing: {str(e)}")
        logger.exception("Stack trace:")