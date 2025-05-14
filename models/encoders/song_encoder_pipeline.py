# run_pipeline.py
import os
import argparse
import logging
import torch
import pandas as pd
import numpy as np
from tqdm import tqdm
from inference.database import SongDatabase

from song_encoder import (
    EssentiaFeatureExtractor,
    CLAPFeatureExtractor,
    LyricsFeatureExtractor,
    SongFeaturePipeline,
    SongEncoder,
    SongAutoencoder,
    train_autoencoder,
    download_songs
)

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)
def build_song_database(self, 
                       csv_file: str, 
                       output_db_path: str = "song_database.pt", 
                       encoder: Optional[nn.Module] = None) -> SongDatabase:
    """
    Process songs and build a searchable database
    
    Args:
        csv_file: Path to CSV file with song data
        output_db_path: Path to save the database
        encoder: Optional encoder to reduce feature dimensionality
        
    Returns:
        SongDatabase object
    """
    # Process songs to get embeddings
    processed_data = self.process_csv(
        csv_file, 
        output_file=None,  # Don't save to CSV
        encoder=encoder, 
        skip_existing=True
    )
    
    # Create database
    database = SongDatabase(output_db_path)
    
    # Add songs to database
    for _, row in processed_data.iterrows():
        shortcode = row['shortcode']
        title = row['music_title']
        artist = row['music_artist']
        embedding = torch.tensor(row['embedding'], dtype=torch.float32)
        link = row['link'] if 'link' in row else ""
        
        # Add song to database
        database.add_song(
            shortcode=shortcode,
            title=title,
            artist=artist,
            embedding=embedding,
            link=link
        )
    
    return database
def parse_args():
    parser = argparse.ArgumentParser(description="Song Encoding Pipeline")
    parser.add_argument("--csv", type=str, required=True, help="Path to CSV file with song data")
    parser.add_argument("--audio-folder", type=str, default="audio", help="Folder for audio files")
    parser.add_argument("--output", type=str, default=None, help="Output CSV file path")
    parser.add_argument("--download", action="store_true", help="Download songs")
    parser.add_argument("--genius-token", type=str, default=None, help="Genius API token for lyrics")
    parser.add_argument("--embedding-dim", type=int, default=128, help="Embedding dimension")
    parser.add_argument("--train-autoencoder", action="store_true", help="Train autoencoder")
    parser.add_argument("--model-path", type=str, default="song_encoder.pt", help="Path to save/load model")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", 
                        help="Device for model training/inference")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Create folders
    os.makedirs(args.audio_folder, exist_ok=True)
    
    # Download songs if requested
    if args.download:
        logger.info("Downloading songs...")
        download_songs(args.csv, args.audio_folder)
    
    # Initialize feature extractors
    logger.info("Initializing feature extractors...")
    essentia_extractor = EssentiaFeatureExtractor(args.audio_folder)
    clap_extractor = CLAPFeatureExtractor(device=args.device)
    lyrics_extractor = LyricsFeatureExtractor(args.genius_token)
    
    # Create feature pipeline
    pipeline = SongFeaturePipeline(
        essentia_extractor,
        clap_extractor,
        lyrics_extractor,
        args.audio_folder
    )
    
    # Process a small sample to determine feature dimensions
    logger.info("Processing sample to determine feature dimensions...")
    pair_data = pd.read_csv(args.csv)
    pair_data = pair_data[
        pair_data['music_artist'].notna() & 
        pair_data['music_title'].notna()
    ]
    
    # Get first valid sample
    sample_row = pair_data.iloc[0]
    artist = sample_row['music_artist'].replace("'", "").strip()
    song_title = sample_row['music_title'].replace("'", "").strip()
    
    sample_features = pipeline.process_song(artist, song_title)
    input_dim = sample_features.shape[0]
    logger.info(f"Feature dimension: {input_dim}")
    
    # Initialize or load encoder
    encoder = None
    if os.path.exists(args.model_path) and not args.train_autoencoder:
        logger.info(f"Loading encoder from {args.model_path}")
        autoencoder = torch.load(args.model_path, map_location=args.device)
        encoder = autoencoder.encoder
    elif args.train_autoencoder:
        logger.info("Training autoencoder will be done after feature extraction")
    else:
        logger.info("No autoencoder will be used")
    
    # Process all songs
    logger.info("Processing all songs...")
    output_data = pipeline.process_csv(
        args.csv, 
        args.output, 
        encoder, 
        skip_existing=True
    )
    
    # Train autoencoder if requested
    if args.train_autoencoder:
        logger.info("Training autoencoder...")
        
        # Extract features from output data
        features_list = [np.array(row['embedding']) for _, row in output_data.iterrows()]
        
        # Train autoencoder
        autoencoder = train_autoencoder(
            features_list,
            hidden_dim=256,
            embedding_dim=args.embedding_dim,
            epochs=100,
            device=args.device
        )
        
        # Save model
        torch.save(autoencoder, args.model_path)
        logger.info(f"Autoencoder saved to {args.model_path}")
        
        # Re-encode features with trained encoder
        logger.info("Re-encoding features with trained encoder...")
        encoder = autoencoder.encoder
        
        # Update embeddings
        for i, row in tqdm(output_data.iterrows(), total=len(output_data), desc="Re-encoding"):
            features = np.array(row['embedding'])
            features_tensor = torch.tensor(features, dtype=torch.float32).unsqueeze(0).to(args.device)
            
            with torch.no_grad():
                encoded_features = encoder(features_tensor).squeeze(0).cpu().numpy()
            
            output_data.at[i, 'embedding'] = encoded_features.tolist()
        
        # Save updated embeddings
        output_data.to_csv(args.output if args.output else "song_embeddings_encoded.csv", index=False)
    
    logger.info("Pipeline completed successfully!")

if __name__ == "__main__":
    main()