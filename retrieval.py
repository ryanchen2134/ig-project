import torch
import pandas as pd
import numpy as np
import os
from PIL import Image
from torchvision import transforms
import torch.nn.functional as F
import ast

def prepare_song_database(csv_path, output_path):
    # Load the CSV
    df = pd.read_csv(csv_path)
    
    # Convert embeddings from string to numpy arrays if needed
    if isinstance(df['audio_embedding'].iloc[0], str):
        df['audio_embedding'] = df['audio_embedding'].apply(lambda x: np.array(ast.literal_eval(x)))
    
    # Create a dictionary to track unique songs
    unique_songs = {}
    unique_song_count = 0
    
    # Create new DataFrame for the song database
    song_data = {
        'id': [],
        'title': [],
        'artist': [],
        'shortcode': [],  # Keep original shortcode for reference
        'link': [],       # Keep link for reference
    }
    
    # Add columns for embedding values
    embedding_dim = len(df['audio_embedding'].iloc[0])
    for i in range(embedding_dim):
        song_data[f'embed_{i}'] = []
    
    # Process each row
    for idx, row in df.iterrows():
        # Create a fingerprint for the song based on its embedding
        embedding_fingerprint = tuple(row['audio_embedding'].round(4))
        
        # Check if this song is already in our unique songs
        if embedding_fingerprint not in unique_songs:
            # New unique song - assign an ID
            song_id = f"SONG_{unique_song_count:04d}"
            unique_songs[embedding_fingerprint] = song_id
            unique_song_count += 1
            
            # Add to song database
            song_data['id'].append(song_id)
            
            # Use the new music_title and music_artist columns
            song_data['title'].append(row['music_title'])
            song_data['artist'].append(row['music_artist'])
            song_data['shortcode'].append(row['shortcode'])
            song_data['link'].append(row['link'])
            
            # Add embedding values
            for i, val in enumerate(row['audio_embedding']):
                song_data[f'embed_{i}'].append(val)
    
    # Create DataFrame
    song_df = pd.DataFrame(song_data)
    
    # Save to CSV
    song_df.to_csv(output_path, index=False)
    
    print(f"Created song database with {len(song_df)} unique songs out of {len(df)} total entries")
    return song_df

def build_song_database(model, song_data_path, save_path, device='cuda'):
    """
    Build and save a database of song embeddings for retrieval
    
    Args:
        model: Trained ContrastiveImageSongModel
        song_data_path: Path to song embedding data
        save_path: Where to save the database
        device: Device to use for processing
    """
    model.eval()
    
    # Load song data
    song_dataset = pd.read_csv(song_data_path)
    
    # Create dictionary to store song info and embeddings
    song_database = {
        'embeddings': [],
        'song_ids': [],
        'titles': [],
        'artists': []
    }
    
    # Process songs in batches
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(song_dataset), batch_size):
            batch = song_dataset.iloc[i:i+batch_size]
            
            # Get embeddings
            embedding_columns = [col for col in song_dataset.columns if col.startswith('embed_')]
            song_features = torch.tensor(batch[embedding_columns].values, dtype=torch.float32).to(device)
            
            # Get projected embeddings using song_only mode
            _, projected_embeddings = model(None, song_features, song_only=True)
            
            # Store embeddings and metadata
            song_database['embeddings'].append(projected_embeddings.cpu())
            song_database['song_ids'].extend(batch['id'].tolist())
            song_database['titles'].extend(batch['title'].tolist())
            song_database['artists'].extend(batch['artist'].tolist())
            
            print(f"Processed {i+len(batch)}/{len(song_dataset)} songs")
    
    # Concatenate all embeddings
    song_database['embeddings'] = torch.cat(song_database['embeddings'], dim=0)
    
    # Save database
    torch.save(song_database, save_path)
    print(f"Song database with {len(song_database['song_ids'])} songs saved to {save_path}")
    
    return song_database

# The rest of the retrieval.py file remains the same
def retrieve_songs_for_image(model, image_path, song_database, top_k=5, device='cuda'):
    """
    Retrieve the top-k songs that match an input image
    
    Args:
        model: Trained ContrastiveImageSongModel
        image_path: Path to the query image
        song_database: Database of song embeddings
        top_k: Number of songs to retrieve
        device: Device to use for processing
    
    Returns:
        List of dictionaries containing song information
    """
    model.eval()
    
    # Load and preprocess the image
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get image embedding
    with torch.no_grad():
        image_embedding, _ = model(image_tensor, None, image_only=True)
    
    # Compute similarities with all songs
    song_embeddings = song_database['embeddings'].to(device)
    similarities = torch.mm(image_embedding, song_embeddings.t()).squeeze()
    
    # Get top-k songs
    top_indices = torch.argsort(similarities, descending=True)[:top_k]
    
    # Prepare results
    results = []
    for idx in top_indices:
        idx = idx.item()
        results.append({
            'song_id': song_database['song_ids'][idx],
            'title': song_database['titles'][idx],
            'artist': song_database['artists'][idx],
            'similarity': similarities[idx].item()
        })
    
    return results