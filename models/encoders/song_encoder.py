# song_encoder.py
import os
import torch
import numpy as np
import pandas as pd
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import essentia.standard as es
import requests
import json
import lyricsgenius
from transformers import AutoProcessor, CLAPModel
from tqdm import tqdm
import logging
from typing import Dict, List, Tuple, Optional, Any, Union

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class SongEncoder(nn.Module):
    """
    Neural network to encode concatenated song features into a compact embedding
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, embedding_dim: int = 128):
        """
        Initialize the song encoder
        
        Args:
            input_dim: Total dimension of concatenated features
            hidden_dim: Dimension of hidden layer
            embedding_dim: Dimension of output embedding
        """
        super(SongEncoder, self).__init__()
        
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, embedding_dim)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        embedding = self.encoder(x)
        # L2 normalization
        return F.normalize(embedding, p=2, dim=1)

class SongAutoencoder(nn.Module):
    """
    Autoencoder for dimensionality reduction and feature learning
    """
    def __init__(self, input_dim: int, hidden_dim: int = 256, embedding_dim: int = 128):
        super(SongAutoencoder, self).__init__()
        
        # Encoder
        self.encoder = SongEncoder(input_dim, hidden_dim, embedding_dim)
        
        # Decoder (reconstructs the input)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, input_dim)
        )
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward pass returning both encoded and decoded data"""
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return encoded, decoded

class SongFeatureDataset(Dataset):
    """Dataset for song features"""
    def __init__(self, features_list: List[np.ndarray]):
        self.features = [torch.tensor(features, dtype=torch.float32) for features in features_list]
    
    def __len__(self) -> int:
        return len(self.features)
    
    def __getitem__(self, idx: int) -> torch.Tensor:
        return self.features[idx]

class EssentiaFeatureExtractor:
    """Extract audio features using Essentia"""
    def __init__(self, audio_folder: str = "audio"):
        self.audio_folder = audio_folder
        os.makedirs(audio_folder, exist_ok=True)
    
    def extract_features(self, artist: str, song_title: str) -> np.ndarray:
        """
        Extract audio features using Essentia
        
        Args:
            artist: Artist name
            song_title: Song title
            
        Returns:
            numpy array of audio features
        """
        audio_file = os.path.join(self.audio_folder, f"{artist} - {song_title}.wav")
        
        if not os.path.exists(audio_file):
            logger.warning(f"Audio file not found: {audio_file}")
            return None
        
        try:
            # Extract features using MusicExtractor
            features, _ = es.MusicExtractor(
                lowlevelStats=['mean', 'stdev'],
                rhythmStats=['mean', 'stdev'],
                tonalStats=['mean', 'stdev']
            )(audio_file)
            
            # Extract specific features
            feature_vector = self._extract_relevant_features(features)
            return feature_vector
            
        except Exception as e:
            logger.error(f"Error extracting features for {artist} - {song_title}: {e}")
            return None
    
    def _extract_relevant_features(self, features: Dict) -> np.ndarray:
        """
        Extract relevant features from Essentia output
        
        Args:
            features: Features dictionary from Essentia
            
        Returns:
            numpy array of selected features
        """
        selected_features = [
            features["lowlevel"]["average_loudness"],
            features["lowlevel"]["dissonance"]["mean"],
            features["lowlevel"]["dynamic_complexity"],
            features["lowlevel"]["spectral_centroid"]["mean"],
            features["lowlevel"]["spectral_flux"]["mean"],
            features["lowlevel"]["spectral_entropy"]["mean"],
            features["lowlevel"]["spectral_rolloff"]["mean"],
            features["lowlevel"]["hfc"]["mean"],
            features["lowlevel"]["pitch_salience"]["mean"],
            features["lowlevel"]["spectral_complexity"]["mean"],
            features["lowlevel"]["spectral_spread"]["mean"],
            features["lowlevel"]["spectral_strongpeak"]["mean"],
            *features["lowlevel"]["barkbands"]["mean"],
            *features["lowlevel"]["mfcc"]["mean"],
            *features["lowlevel"]["gfcc"]["mean"],
            features["rhythm"]["bpm"],
            features["rhythm"]["danceability"],
            features["rhythm"]["onset_rate"],
            features["tonal"]["chords_strength"]["mean"],
            features["tonal"]["hpcp_crest"]["mean"],
            features["tonal"]["hpcp_entropy"]["mean"],
            features["tonal"]["key_krumhansl"]["strength"]
        ]
        
        return np.array(selected_features, dtype=np.float32)

class CLAPFeatureExtractor:
    """Extract audio embeddings using CLAP model"""
    def __init__(self, 
                 model_name: str = "laion/clap-htsat-fused", 
                 device: str = "cuda" if torch.cuda.is_available() else "cpu"):
        self.device = device
        logger.info(f"Loading CLAP model: {model_name} on {device}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = CLAPModel.from_pretrained(model_name).to(device)
        
    def extract_features(self, artist: str, song_title: str, audio_folder: str = "audio") -> np.ndarray:
        """
        Extract audio embeddings using CLAP
        
        Args:
            artist: Artist name
            song_title: Song title
            audio_folder: Folder containing audio files
            
        Returns:
            numpy array of CLAP embeddings
        """
        audio_file = os.path.join(audio_folder, f"{artist} - {song_title}.wav")
        
        if not os.path.exists(audio_file):
            logger.warning(f"Audio file not found: {audio_file}")
            return None
        
        try:
            # Load and process audio
            waveform, sample_rate = es.AudioLoader(filename=audio_file)()
            
            # Convert to mono and resample if needed
            if len(waveform.shape) > 1:
                waveform = np.mean(waveform, axis=1)
            
            # Process with CLAP
            inputs = self.processor(
                audios=waveform, 
                sampling_rate=sample_rate, 
                return_tensors="pt"
            ).to(self.device)
            
            # Get audio embeddings
            with torch.no_grad():
                audio_embedding = self.model.get_audio_features(**inputs)
                
            # Return embeddings as numpy array
            return audio_embedding.cpu().squeeze().numpy()
            
        except Exception as e:
            logger.error(f"Error extracting CLAP features for {artist} - {song_title}: {e}")
            return None

class LyricsFeatureExtractor:
    """Extract and process song lyrics"""
    def __init__(self, genius_api_token: Optional[str] = None):
        self.genius = None
        if genius_api_token:
            self.genius = lyricsgenius.Genius(genius_api_token)
            self.genius.verbose = False
    
    def get_lyrics(self, artist: str, song_title: str) -> str:
        """
        Get lyrics for a song
        
        Args:
            artist: Artist name
            song_title: Song title
            
        Returns:
            Song lyrics as string or None if not found
        """
        if not self.genius:
            logger.warning("Genius API token not provided, skipping lyrics extraction")
            return None
        
        try:
            song = self.genius.search_song(song_title, artist)
            if song:
                return song.lyrics
            else:
                logger.warning(f"Lyrics not found for {artist} - {song_title}")
                return None
        except Exception as e:
            logger.error(f"Error getting lyrics for {artist} - {song_title}: {e}")
            return None
    
    def extract_features(self, lyrics: Optional[str]) -> np.ndarray:
        """
        Extract features from lyrics
        
        Args:
            lyrics: Song lyrics as string
            
        Returns:
            numpy array of lyrics features
        """
        if not lyrics:
            # Return zeros if no lyrics found
            return np.zeros(20, dtype=np.float32)
        
        # Basic lyrics features
        features = []
        
        # Clean lyrics
        lyrics = lyrics.replace('\n', ' ').lower()
        
        # Word count
        words = lyrics.split()
        word_count = len(words)
        features.append(min(word_count / 500, 1.0))  # Normalized word count
        
        # Get simple sentiment using positive and negative word lists
        positive_words = ['love', 'happy', 'joy', 'sweet', 'good', 'best', 'beautiful', 'amazing', 'wonderful']
        negative_words = ['hate', 'sad', 'pain', 'hurt', 'bad', 'worst', 'terrible', 'awful', 'wrong']
        
        positive_count = sum(1 for word in words if word in positive_words)
        negative_count = sum(1 for word in words if word in negative_words)
        
        # Simple sentiment score
        if word_count > 0:
            features.append(positive_count / word_count)
            features.append(negative_count / word_count)
            features.append((positive_count - negative_count) / word_count)  # Net sentiment
        else:
            features.extend([0, 0, 0])
        
        # Repetition features
        unique_words = len(set(words))
        if word_count > 0:
            features.append(unique_words / word_count)  # Lexical diversity
        else:
            features.append(0)
        
        # Add more basic NLP features here
        
        # Pad to fixed size
        padded_features = features + [0] * (20 - len(features))
        
        return np.array(padded_features, dtype=np.float32)

class SongFeaturePipeline:
    """Pipeline to extract and combine features from multiple sources"""
    def __init__(self, 
                 essentia_extractor: EssentiaFeatureExtractor,
                 clap_extractor: CLAPFeatureExtractor,
                 lyrics_extractor: LyricsFeatureExtractor,
                 audio_folder: str = "audio"):
        self.essentia_extractor = essentia_extractor
        self.clap_extractor = clap_extractor
        self.lyrics_extractor = lyrics_extractor
        self.audio_folder = audio_folder
    
    def process_song(self, artist: str, song_title: str) -> np.ndarray:
        """
        Process a single song to extract all features
        
        Args:
            artist: Artist name
            song_title: Song title
            
        Returns:
            Combined feature vector
        """
        # Get Essentia features
        essentia_features = self.essentia_extractor.extract_features(artist, song_title)
        if essentia_features is None:
            logger.error(f"Failed to extract Essentia features for {artist} - {song_title}")
            return None
        
        # Get CLAP embeddings
        clap_features = self.clap_extractor.extract_features(artist, song_title, self.audio_folder)
        if clap_features is None:
            logger.error(f"Failed to extract CLAP features for {artist} - {song_title}")
            return None
        
        # Get lyrics and features
        lyrics = self.lyrics_extractor.get_lyrics(artist, song_title)
        lyrics_features = self.lyrics_extractor.extract_features(lyrics)
        
        # Combine all features
        combined_features = np.concatenate([
            essentia_features, 
            clap_features, 
            lyrics_features
        ])
        
        return combined_features
    
    def process_csv(self, 
                   csv_file: str, 
                   output_file: Optional[str] = None,
                   encoder: Optional[nn.Module] = None,
                   skip_existing: bool = True) -> pd.DataFrame:
        """
        Process songs listed in a CSV file
        
        Args:
            csv_file: Path to CSV file with song data
            output_file: Path to save output CSV
            encoder: Optional encoder to reduce feature dimensionality
            skip_existing: Whether to skip already processed songs
            
        Returns:
            DataFrame with song data and embeddings
        """
        # Read CSV
        pair_data = pd.read_csv(csv_file)
        
        # Filter out rows with NaN values in required columns
        pair_data = pair_data[
            pair_data['music_artist'].notna() & 
            pair_data['music_title'].notna()
        ]
        
        # Set default output file if not provided
        if output_file is None:
            output_file = os.path.join(os.path.dirname(csv_file), "song_embeddings.csv")
        
        # Load existing data if available and skip_existing is True
        existing_data = {}
        if skip_existing and os.path.exists(output_file):
            existing_df = pd.read_csv(output_file)
            existing_data = {row['shortcode']: row for _, row in existing_df.iterrows()}
        
        # Prepare output data
        output_data = []
        
        # Process each song
        for index, row in tqdm(pair_data.iterrows(), total=len(pair_data), desc="Processing songs"):
            try:
                shortcode = row['shortcode']
                
                # Skip if already processed
                if skip_existing and shortcode in existing_data:
                    logger.info(f"Skipping already processed song: {shortcode}")
                    output_data.append(existing_data[shortcode])
                    continue
                
                # Clean artist and song name
                artist = row['music_artist'].replace("'", "").strip()
                song_title = row['music_title'].replace("'", "").strip()
                
                logger.info(f"Processing {index + 1}/{len(pair_data)}: {artist} - {song_title}")
                
                # Extract combined features
                combined_features = self.process_song(artist, song_title)
                
                if combined_features is None:
                    logger.warning(f"Skipping {artist} - {song_title} due to missing features")
                    continue
                
                # Apply encoder if provided
                if encoder is not None:
                    with torch.no_grad():
                        # Convert to tensor, add batch dimension
                        features_tensor = torch.tensor(combined_features, dtype=torch.float32).unsqueeze(0)
                        encoded_features = encoder(features_tensor).squeeze(0).numpy()
                else:
                    encoded_features = combined_features
                
                # Create output row
                output_row = {
                    'shortcode': shortcode,
                    'link': row['head_image_url'] if 'head_image_url' in row else None,
                    'music_artist': artist,
                    'music_title': song_title,
                    'embedding': encoded_features.tolist()
                }
                
                output_data.append(output_row)
                
                # Save after each song to avoid losing progress
                pd.DataFrame(output_data).to_csv(output_file, index=False)
                
            except Exception as e:
                logger.error(f"Error processing {row['music_artist']} - {row['music_title']}: {e}")
        
        return pd.DataFrame(output_data)

def train_autoencoder(features_list: List[np.ndarray], 
                      hidden_dim: int = 256,
                      embedding_dim: int = 128,
                      batch_size: int = 32,
                      epochs: int = 100,
                      learning_rate: float = 1e-3,
                      device: str = "cuda" if torch.cuda.is_available() else "cpu") -> SongAutoencoder:
    """
    Train an autoencoder for dimensionality reduction
    
    Args:
        features_list: List of feature arrays
        hidden_dim: Hidden layer dimension
        embedding_dim: Embedding dimension
        batch_size: Training batch size
        epochs: Number of training epochs
        learning_rate: Learning rate
        device: Training device
        
    Returns:
        Trained autoencoder
    """
    # Create dataset
    dataset = SongFeatureDataset(features_list)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Create model
    input_dim = features_list[0].shape[0]
    model = SongAutoencoder(input_dim, hidden_dim, embedding_dim).to(device)
    
    # Optimizer and loss
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    # Training loop
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}"):
            batch = batch.to(device)
            
            # Forward pass
            _, reconstructed = model(batch)
            loss = criterion(reconstructed, batch)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
        
        avg_loss = total_loss / len(dataloader)
        logger.info(f"Epoch {epoch+1}/{epochs}: Loss = {avg_loss:.6f}")
    
    return model

def download_songs(csv_file: str, audio_folder: str = "audio"):
    """
    Download songs using spotdl
    
    Args:
        csv_file: Path to CSV file with song data
        audio_folder: Folder to save downloaded audio
    """
    os.makedirs(audio_folder, exist_ok=True)
    
    # Read CSV
    pair_data = pd.read_csv(csv_file)
    
    # Filter out rows with NaN values
    pair_data = pair_data[
        pair_data['music_artist'].notna() & 
        pair_data['music_title'].notna()
    ]
    
    # Download each song
    for index, row in tqdm(pair_data.iterrows(), total=len(pair_data), desc="Downloading songs"):
        try:
            # Clean artist and song name
            artist = row['music_artist'].replace("'", "").strip()
            song_title = row['music_title'].replace("'", "").strip()
            
            # Check if file already exists
            expected_file = os.path.join(audio_folder, f"{artist} - {song_title}.wav")
            if os.path.exists(expected_file):
                logger.info(f"Song already downloaded: {expected_file}")
                continue
            
            logger.info(f"Downloading {index + 1}/{len(pair_data)}: {artist} - {song_title}")
            
            # spotdl format
            download_name = os.path.join(audio_folder, f"{artist} - {song_title}") + ".{output-ext}"
            os.system(f"spotdl download '{artist} - {song_title}' --format wav --output '{download_name}'")
            
        except Exception as e:
            logger.error(f"Failed to download {artist} - {song_title}: {e}")