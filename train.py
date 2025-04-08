# train.py
import torch
from torch.utils.data import DataLoader, random_split
import torch.optim as optim
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import your model and dataset
from models.decoder.contrastive import ContrastiveImageSongModel, NTXentLoss
from data.dloader import ImgSongDataset

def train_contrastive_model(model, train_loader, val_loader, optimizer, loss_fn, 
                           num_epochs=50, patience=10, device='cuda'):
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
    
    Returns:
        model: Trained model
        history: Training history
    """
    model = model.to(device)
    best_val_loss = float('inf')
    best_model = None
    no_improve_count = 0
    
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        
        for batch_idx, (images, song_embeddings) in enumerate(train_loader):  # Renamed variable
            images = images.to(device)
            song_embeddings = song_embeddings.to(device)  # Renamed variable
            
            optimizer.zero_grad()
            
            # Forward pass
            image_embeddings, projected_song_embeddings = model(images, song_embeddings)  # Updated variable name
            
            # Calculate loss
            loss = loss_fn(image_embeddings, projected_song_embeddings)  # Updated variable name
            
            # Check for NaN/Inf
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: Training loss is {loss.item()}, stopping training")
                return model, {'train_loss': train_losses, 'val_loss': val_losses}  # Added return to actually stop training
            
            # Backward pass and optimize
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}], Batch [{batch_idx+1}/{len(train_loader)}], Loss: {loss.item():.4f}")
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, song_embeddings in val_loader:  # Renamed variable
                images = images.to(device)
                song_embeddings = song_embeddings.to(device)  # Renamed variable
                
                # Forward pass
                image_embeddings, projected_song_embeddings = model(images, song_embeddings)  # Updated variable name
                
                # Calculate loss
                loss = loss_fn(image_embeddings, projected_song_embeddings)  # Updated variable name
                
                if torch.isnan(loss) or torch.isinf(loss):
                    print(f"Warning: Validation loss is {loss.item()}, stopping training")
                    return model, {'train_loss': train_losses, 'val_loss': val_losses}  # Added return to actually stop training
                
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        
        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_model = model.state_dict().copy()
            no_improve_count = 0
            print(f"Validation loss improved to {avg_val_loss:.4f}")
        else:
            no_improve_count += 1
            print(f"No improvement for {no_improve_count} epochs")
            
            if no_improve_count >= patience:
                print(f"Early stopping triggered after {epoch+1} epochs")
                break
    
    # Load best model
    model.load_state_dict(best_model)
    
    history = {
        'train_loss': train_losses,
        'val_loss': val_losses,
        'best_val_loss': best_val_loss
    }
    
    return model, history

def plot_training_history(history):
    """Plot training and validation loss"""
    plt.figure(figsize=(10, 6))
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Contrastive Learning Training History')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.savefig('contrastive_training_history.png')
    plt.show()

if __name__ == "__main__":
    # Set random seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)

    # Parameters
    batch_size = 32
    embedding_dim = 128
    learning_rate = 1e-4
    weight_decay = 1e-5
    num_epochs = 100
    patience = 15
    temperature = 0.07  # Temperature parameter for NT-Xent loss
    
    # Device configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Define your data paths here
    data_path = "DISCO/song_csv/Lady Gaga-Die with a smile.csv"  # Update this
    img_folder = "models/decoder/test_img"
    
    # Create dataset
    dataset = ImgSongDataset(data_path=data_path, img_folder=img_folder)
    print(f"Dataset size: {len(dataset)}")
    
    # Get song feature dimension from the first item
    _, first_song_features = dataset[0]
    song_embedding_dim = first_song_embedding.shape[0]
    print(f"Song feature dimension: {song_embedding_dim}")
    
    # Split dataset
    train_data, temp_data = train_test_split(dataset, test_size=0.2, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
    
    # Create data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = ContrastiveImageSongModel(song_embedding_dim=song_embedding_dim, embedding_dim=embedding_dim)
    
    # Loss function and optimizer
    loss_fn = NTXentLoss(temperature=temperature, batch_size=batch_size)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Train the model
    trained_model, history = train_contrastive_model(
        model, train_loader, val_loader, optimizer, loss_fn, 
        num_epochs=num_epochs, patience=patience, device=device
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save the model
    torch.save(trained_model.state_dict(), 'contrastive_image_song_model.pth')
    
    print("Training completed and model saved!")

