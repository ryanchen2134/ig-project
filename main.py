"""
Main entry point for training and testing the image-song contrastive model
"""
import argparse
import torch
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
import numpy as np
import logging
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Import models
from models.contrastive.model import ContrastiveImageSongModel
from models.contrastive.loss import NTXentLoss

# Import dataset
from data.dataset import ImgSongDataset

# Import training utilities
from training.train import train_contrastive_model
from training.scheduler import WarmupCosineScheduler

# Import inference utilities
from inference.retrieval import retrieve_songs_for_image, build_song_database, prepare_song_database

# Import utilities
from utils.metrics import calculate_recall_at_k

# Import configuration
import config

def setup_logger(log_dir=None):
    """Set up logging for the training process"""
    if log_dir is None:
        log_dir = config.LOG_DIR
    
    os.makedirs(log_dir, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger('contrastive_training')
    logger.setLevel(logging.INFO)
    
    # Clear existing handlers if they exist
    if logger.hasHandlers():
        logger.handlers.clear()
    
    # Create formatters
    file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    console_formatter = logging.Formatter('%(levelname)s: %(message)s')
    
    # Log file
    log_file = os.path.join(log_dir, f'training_{os.path.basename(config.TRAINED_MODEL_PATH)}.log')
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(file_formatter)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(console_formatter)
    
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def plot_training_history(history, save_path=None):
    """Plot and save training history"""
    plt.figure(figsize=(12, 10))
    
    # Plot training and validation loss
    plt.subplot(2, 2, 1)
    plt.plot(history['train_loss'], label='Training Loss')
    plt.plot(history['val_loss'], label='Validation Loss')
    plt.axvline(x=history.get('best_epoch', 0) - 1, color='r', linestyle='--', label='Best Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Contrastive Learning Loss')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot loss on log scale
    plt.subplot(2, 2, 2)
    plt.semilogy(history['train_loss'], label='Training Loss')
    plt.semilogy(history['val_loss'], label='Validation Loss')
    plt.axvline(x=history.get('best_epoch', 0) - 1, color='r', linestyle='--', label='Best Model')
    plt.xlabel('Epochs')
    plt.ylabel('Loss (log scale)')
    plt.title('Loss (Logarithmic Scale)')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot learning rate
    if 'learning_rates' in history:
        plt.subplot(2, 2, 3)
        plt.plot(history['learning_rates'])
        plt.xlabel('Epochs')
        plt.ylabel('Learning Rate')
        plt.title('Learning Rate Schedule')
        plt.grid(True, linestyle='--', alpha=0.7)
    
    # Plot recall metrics if available
    if 'recalls' in history:
        plt.subplot(2, 2, 4)
        epochs = list(range(len(history['recalls'])))
        r1 = [r['r@1'] for r in history['recalls']]
        r5 = [r['r@5'] for r in history['recalls']]
        r10 = [r['r@10'] for r in history['recalls']]
        
        plt.plot(epochs, r1, 'o-', label='R@1')
        plt.plot(epochs, r5, 's-', label='R@5')
        plt.plot(epochs, r10, '^-', label='R@10')
        plt.xlabel('Epochs')
        plt.ylabel('Recall (%)')
        plt.title('Retrieval Performance')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    
    if save_path is None:
        save_path = os.path.join(config.CHECKPOINT_DIR, 'training_history.png')
    
    plt.savefig(save_path)
    plt.close()
    
    return save_path

def train(args):
    """
    Train the contrastive model
    
    Args:
        args: Command line arguments
    """
    logger = setup_logger()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Log parameters
    logger.info("=== Training Parameters ===")
    logger.info(f"Model: CLIP-based with {args.clip_model} backbone")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Embedding dimension: {args.embedding_dim}")
    logger.info(f"Learning rate: {args.lr}")
    logger.info(f"Weight decay: {args.weight_decay}")
    logger.info(f"Max epochs: {args.epochs}")
    logger.info(f"Patience: {args.patience}")
    logger.info(f"Temperature: {args.temperature}")
    logger.info(f"Hard negative weight: {args.hard_negative_weight}")
    logger.info(f"Mixup alpha: {args.mixup_alpha}")
    logger.info(f"Device: {config.DEVICE}")
    
    # Create dataset
    if args.clip_model:
        import clip
        _, preprocess = clip.load(args.clip_model, device="cpu")
    
    dataset = ImgSongDataset(
        file_path=args.data_path,
        img_folder=args.img_folder,
        transform=preprocess if args.clip_model else None
    )
    logger.info(f"Dataset loaded with {len(dataset)} samples")
    
    # Get song embedding dimension
    _, first_song_features = dataset[0]
    song_embedding_dim = first_song_features.shape[0]
    logger.info(f"Song embedding dimension: {song_embedding_dim}")
    
    # Split dataset
    train_size = int(len(dataset) * (1 - args.test_split))
    val_size = int((len(dataset) - train_size) * 0.5)
    test_size = len(dataset) - train_size - val_size
    
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(args.seed)
    )
    
    logger.info(f"Train set: {len(train_dataset)} samples")
    logger.info(f"Validation set: {len(val_dataset)} samples")
    logger.info(f"Test set: {len(test_dataset)} samples")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers
    )
    
    # Initialize model
    model = ContrastiveImageSongModel(
        song_embedding_dim=song_embedding_dim,
        embedding_dim=args.embedding_dim,
        clip_model_name=args.clip_model,
        freeze_image_encoder=args.freeze_base
    )
    logger.info(f"Model initialized")
    
    # Loss function
    loss_fn = NTXentLoss(
        temperature=args.temperature,
        hard_negative_weight=args.hard_negative_weight
    )
    
    # Optimizer - use different learning rates for different parts of the model
    if args.freeze_base:
        # Only optimize the projection layers
        optimizer = optim.AdamW([
            {'params': model.song_projection.parameters()},
            {'params': model.image_encoder.projection.parameters()}
        ], lr=args.lr, weight_decay=args.weight_decay)
    else:
        # Optimize everything with different learning rates
        optimizer = optim.AdamW([
            {'params': model.song_projection.parameters(), 'lr': args.lr},
            {'params': model.image_encoder.projection.parameters(), 'lr': args.lr},
            {'params': model.image_encoder.clip_model.parameters(), 'lr': args.lr * 0.1}
        ], lr=args.lr, weight_decay=args.weight_decay)
    
    # Learning rate scheduler
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_epochs=args.warmup_epochs,
        max_epochs=args.epochs,
        warmup_start_lr=args.lr * 0.1,
        base_lr=args.lr,
        min_lr=args.lr * 0.01
    )
    
    logger.info("Starting training...")
    model, history = train_contrastive_model(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        num_epochs=args.epochs,
        patience=args.patience,
        device=config.DEVICE,
        checkpoint_dir=config.CHECKPOINT_DIR,
        visualize_every=5,
        logger=logger,
        scheduler=scheduler,
        mixup_alpha=args.mixup_alpha
    )
    
    # Plot training history
    plot_training_history(history)
    
    # Save the model
    torch.save({
        'model_state_dict': model.state_dict(),
        'embedding_dim': args.embedding_dim,
        'clip_model': args.clip_model,
        'song_embedding_dim': song_embedding_dim
    }, args.model_path)
    
    logger.info(f"Model saved to {args.model_path}")
    
    # Evaluate on test set
    logger.info("Evaluating on test set...")
    model.eval()
    
    with torch.no_grad():
        all_image_embs = []
        all_song_embs = []
        
        for images, song_features in test_loader:
            images = images.to(config.DEVICE)
            song_features = song_features.to(config.DEVICE)
            
            image_emb, song_emb = model(images, song_features)
            all_image_embs.append(image_emb.cpu())
            all_song_embs.append(song_emb.cpu())
        
        # Concatenate all embeddings
        all_image_embs = torch.cat(all_image_embs, dim=0)
        all_song_embs = torch.cat(all_song_embs, dim=0)
        
        # Calculate recall metrics
        recall_metrics = calculate_recall_at_k(all_image_embs, all_song_embs)
        
        for k, v in recall_metrics.items():
            logger.info(f"{k}: {v:.2f}%")
    
    logger.info("Training completed successfully!")

def create_song_database(args):
    """
    Create a song database for retrieval
    
    Args:
        args: Command line arguments
    """
    logger = setup_logger()
    
    # Load model
    logger.info(f"Loading model from {args.model_path}")
    
    checkpoint = torch.load(args.model_path, map_location=config.DEVICE)
    
    # Extract model parameters
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        embedding_dim = checkpoint.get('embedding_dim', config.DEFAULT_EMBEDDING_DIM)
        clip_model = checkpoint.get('clip_model', config.DEFAULT_CLIP_MODEL)
        song_embedding_dim = checkpoint.get('song_embedding_dim', None)
    else:
        logger.error("Invalid model checkpoint format")
        return
    
    # Prepare song database from original CSV
    if args.input_csv:
        logger.info(f"Preparing song database from {args.input_csv}")
        
        prepared_csv_path = os.path.join(config.DATA_DIR, "prepared_song_data.csv")
        song_df = prepare_song_database(args.input_csv, prepared_csv_path)
        
        if song_embedding_dim is None:
            # Determine song embedding dimension from the prepared CSV
            embedding_cols = [col for col in song_df.columns if col.startswith('embed_')]
            song_embedding_dim = len(embedding_cols)
        
        logger.info(f"Song data prepared with {len(song_df)} songs")
    else:
        logger.error("Input CSV required to create song database")
        return
    
    # Initialize model
    model = ContrastiveImageSongModel(
        song_embedding_dim=song_embedding_dim,
        embedding_dim=embedding_dim,
        clip_model_name=clip_model,
        freeze_image_encoder=True  # Not training, just inference
    ).to(config.DEVICE)
    
    model.load_state_dict(model_state)
    model.eval()
    
    # Build song database with projected embeddings
    logger.info("Building song database with model projections")
    song_db = build_song_database(
        model=model,
        song_data_path=prepared_csv_path,
        save_path=args.output_db,
        device=config.DEVICE
    )
    
    logger.info(f"Song database created with {len(song_db['song_ids'])} songs")
    logger.info(f"Saved to {args.output_db}")

def test_retrieval(args):
    """
    Test retrieval with a sample image
    
    Args:
        args: Command line arguments
    """
    if not os.path.exists(args.model_path):
        print(f"Model not found at {args.model_path}")
        return
    
    if not os.path.exists(args.song_db):
        print(f"Song database not found at {args.song_db}")
        return
    
    if not os.path.exists(args.test_image):
        print(f"Test image not found at {args.test_image}")
        return
    
    # Load model
    print(f"Loading model from {args.model_path}")
    
    checkpoint = torch.load(args.model_path, map_location=config.DEVICE)
    
    # Extract model parameters
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        embedding_dim = checkpoint.get('embedding_dim', config.DEFAULT_EMBEDDING_DIM)
        clip_model = checkpoint.get('clip_model', config.DEFAULT_CLIP_MODEL)
        song_embedding_dim = checkpoint.get('song_embedding_dim', None)
    else:
        print("Invalid model checkpoint format")
        return
    
    # Load song database
    print(f"Loading song database from {args.song_db}")
    song_db = torch.load(args.song_db, map_location=config.DEVICE)
    
    if song_embedding_dim is None:
        # Try to determine from the song database
        song_embedding_dim = song_db['embeddings'].size(1)
    
    # Initialize model
    model = ContrastiveImageSongModel(
        song_embedding_dim=song_embedding_dim,
        embedding_dim=embedding_dim,
        clip_model_name=clip_model,
        freeze_image_encoder=True  # Just for inference
    ).to(config.DEVICE)
    
    model.load_state_dict(model_state)
    model.eval()
    
    # Show the test image
    from PIL import Image
    import matplotlib.pyplot as plt
    
    img = Image.open(args.test_image).convert('RGB')
    plt.figure(figsize=(6, 6))
    plt.imshow(img)
    plt.title("Query Image")
    plt.axis('off')
    plt.show()
    
    # Retrieve matching songs
    results = retrieve_songs_for_image(
        model=model,
        image_path=args.test_image,
        song_database=song_db,
        top_k=args.top_k,
        device=config.DEVICE
    )
    
    # Print results
    print("\nTop matching songs:")
    for i, song in enumerate(results):
        print(f"{i+1}. {song['title']} by {song['artist']} (similarity: {song['similarity']:.4f})")
    
    print("\nRetrieval test completed successfully!")

def create_song_database(args):
    """
    Create a song database for retrieval
    
    Args:
        args: Command line arguments
    """
    logger = setup_logger()
    
    # Load or train encoder
    if os.path.exists(args.encoder_path):
        logger.info(f"Loading encoder from {args.encoder_path}")
        autoencoder = torch.load(args.encoder_path, map_location=config.DEVICE)
        encoder = autoencoder.encoder
    else:
        logger.error(f"Encoder not found at {args.encoder_path}")
        return
    
    # Import song encoder pipeline
    from models.encoders.song_encoder_pipeline import (
        EssentiaFeatureExtractor,
        CLAPFeatureExtractor,
        LyricsFeatureExtractor,
        SongFeaturePipeline
    )
    
    # Initialize feature extractors
    logger.info("Initializing feature extractors...")
    essentia_extractor = EssentiaFeatureExtractor(args.audio_folder)
    clap_extractor = CLAPFeatureExtractor(device=config.DEVICE)
    lyrics_extractor = LyricsFeatureExtractor(args.genius_token)
    
    # Create feature pipeline
    pipeline = SongFeaturePipeline(
        essentia_extractor,
        clap_extractor,
        lyrics_extractor,
        args.audio_folder
    )
    
    # Build song database
    logger.info(f"Building song database from {args.data_path}...")
    database = pipeline.build_song_database(
        args.data_path,
        args.song_db_path,
        encoder
    )
    
    logger.info(f"Created song database with {len(database)} songs")
    
def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Image-Song Contrastive Learning')
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train the model')
    train_parser.add_argument('--data_path', type=str, required=True, help='Path to CSV data file')
    train_parser.add_argument('--img_folder', type=str, required=True, help='Path to folder with images')
    train_parser.add_argument('--model_path', type=str, default=config.TRAINED_MODEL_PATH, help='Path to save model')
    train_parser.add_argument('--clip_model', type=str, default=config.DEFAULT_CLIP_MODEL, help='CLIP model variant')
    train_parser.add_argument('--embedding_dim', type=int, default=config.DEFAULT_EMBEDDING_DIM, help='Embedding dimension')
    train_parser.add_argument('--batch_size', type=int, default=config.BATCH_SIZE, help='Batch size')
    train_parser.add_argument('--lr', type=float, default=config.LEARNING_RATE, help='Learning rate')
    train_parser.add_argument('--weight_decay', type=float, default=config.WEIGHT_DECAY, help='Weight decay')
    train_parser.add_argument('--epochs', type=int, default=config.NUM_EPOCHS, help='Number of epochs')
    train_parser.add_argument('--patience', type=int, default=config.PATIENCE, help='Early stopping patience')
    train_parser.add_argument('--temperature', type=float, default=config.TEMPERATURE, help='Temperature for NT-Xent loss')
    train_parser.add_argument('--hard_negative_weight', type=float, default=config.HARD_NEGATIVE_WEIGHT, help='Hard negative weight')
    train_parser.add_argument('--mixup_alpha', type=float, default=config.MIXUP_ALPHA, help='Mixup alpha parameter')
    train_parser.add_argument('--test_split', type=float, default=config.TRAIN_TEST_SPLIT, help='Test split ratio')
    train_parser.add_argument('--workers', type=int, default=4, help='Number of dataloader workers')
    train_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    train_parser.add_argument('--freeze_base', action='store_true', help='Freeze CLIP base model')
    train_parser.add_argument('--warmup_epochs', type=int, default=5, help='Warmup epochs for scheduler')
    
    # Create song database command
    db_parser = subparsers.add_parser('create_db', help='Create song database for retrieval')
    db_parser.add_argument('--model_path', type=str, default=config.TRAINED_MODEL_PATH, help='Path to trained model')
    db_parser.add_argument('--input_csv', type=str, required=True, help='Path to input CSV with raw song data')
    db_parser.add_argument('--output_db', type=str, default=config.SONG_DATABASE_PATH, help='Path to save song database')
    
    # Test retrieval command
    test_parser = subparsers.add_parser('test', help='Test retrieval with a sample image')
    test_parser.add_argument('--model_path', type=str, default=config.TRAINED_MODEL_PATH, help='Path to trained model')
    test_parser.add_argument('--song_db', type=str, default=config.SONG_DATABASE_PATH, help='Path to song database')
    test_parser.add_argument('--test_image', type=str, required=True, help='Path to test image')
    test_parser.add_argument('--top_k', type=int, default=5, help='Number of top songs to retrieve')
    
    # Song database command
    db_parser = subparsers.add_parser('create_db', help='Create song database for retrieval')
    db_parser.add_argument('--data-path', type=str, required=True, help='Path to CSV with song data')
    db_parser.add_argument('--audio-folder', type=str, default="data/dataset/audio", help='Path to audio folder')
    db_parser.add_argument('--encoder-path', type=str, default="song_autoencoder.pt", help='Path to trained encoder')
    db_parser.add_argument('--song-db-path', type=str, default=config.SONG_DATABASE_PATH, help='Path to save song database')
    db_parser.add_argument('--genius-token', type=str, default=None, help='Genius API token for lyrics')

    
    # Parse arguments
    args = parser.parse_args()
    
    # Execute the appropriate command
    if args.command == 'train':
        train(args)
    elif args.command == 'create_db':
        create_song_database(args)
    elif args.command == 'test':
        test_retrieval(args)
    else:
        parser.print_help()

if __name__ == '__main__':
    main()