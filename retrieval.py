# retrieval.py
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
    if isinstance(df['embedding'].iloc[0], str):
        df['embedding'] = df['embedding'].apply(lambda x: np.array(ast.literal_eval(x)))
    
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
    embedding_dim = len(df['embedding'].iloc[0])
    for i in range(embedding_dim):
        song_data[f'embed_{i}'] = []
    
    # Process each row
    for idx, row in df.iterrows():
        # Create a fingerprint for the song based on its embedding
        # (You might need a more sophisticated method depending on your data)
        embedding_fingerprint = tuple(row['embedding'].round(4))
        
        # Check if this song is already in our unique songs
        if embedding_fingerprint not in unique_songs:
            # New unique song - assign an ID
            song_id = f"SONG_{unique_song_count:04d}"
            unique_songs[embedding_fingerprint] = song_id
            unique_song_count += 1
            
            # Add to song database
            song_data['id'].append(song_id)
            
            # Extract title/artist if possible from link, or use placeholder
            # This is a simplistic approach - adjust based on your link format
            link_parts = row['link'].split('/')
            if len(link_parts) > 0:
                title = link_parts[-1].replace('-', ' ').title()
                artist = "Unknown"  # You might need to extract this differently
            else:
                title = f"Song {song_id}"
                artist = "Unknown"
                
            song_data['title'].append(title)
            song_data['artist'].append(artist)
            song_data['shortcode'].append(row['shortcode'])
            song_data['link'].append(row['link'])
            
            # Add embedding values
            for i, val in enumerate(row['embedding']):
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
    
    # Load song data (adapt this to your data format)
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
            
            # Get embeddings (adapt this to your data format)
            # Modify these column names to match your dataset
            embedding_columns = [col for col in song_dataset.columns if col.startswith('embed_')]
            song_features = torch.tensor(batch[embedding_columns].values, dtype=torch.float32).to(device)
            
            # Get projected embeddings using song_only mode
            _, projected_embeddings = model(None, song_features, song_only=True)
            
            # Store embeddings and metadata
            song_database['embeddings'].append(projected_embeddings.cpu())
            song_database['song_ids'].extend(batch['id'].tolist())  # Adjust column name if needed
            song_database['titles'].extend(batch['title'].tolist())  # Adjust column name if needed
            song_database['artists'].extend(batch['artist'].tolist())  # Adjust column name if needed
            
            print(f"Processed {i+len(batch)}/{len(song_dataset)} songs")
    
    # Concatenate all embeddings
    song_database['embeddings'] = torch.cat(song_database['embeddings'], dim=0)
    
    # Save database
    torch.save(song_database, save_path)
    print(f"Song database with {len(song_database['song_ids'])} songs saved to {save_path}")
    
    return song_database

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

# Usage example (as a script)
if __name__ == "__main__":
    import argparse
    from models.decoder.contrastive import ContrastiveImageSongModel
    
    parser = argparse.ArgumentParser(description='Build song database or retrieve songs')
    parser.add_argument('--mode', type=str, required=True, choices=['build', 'retrieve'],
                       help='Mode: build database or retrieve songs')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model')
    parser.add_argument('--song_data', type=str, 
                       help='Path to song data CSV (for build mode)')
    parser.add_argument('--database_path', type=str, default='song_database.pt',
                       help='Path to save/load the song database')
    parser.add_argument('--image_path', type=str,
                       help='Path to query image (for retrieve mode)')
    parser.add_argument('--top_k', type=int, default=5,
                       help='Number of songs to retrieve (for retrieve mode)')
    
    args = parser.parse_args()
    
    # Determine device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load model
    checkpoint = torch.load(args.model_path, map_location=device)
    
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        # Load from checkpoint dictionary
        model_state = checkpoint['model_state_dict']
        embedding_dim = checkpoint.get('embedding_dim', 128)
        # You need to determine song_embedding_dim somehow
        song_embedding_dim = 128  # Default, should be determined from data
    else:
        # Direct state dict
        model_state = checkpoint
        embedding_dim = 128  # Default
        song_embedding_dim = 128  # Default
    
    # Initialize model
    model = ContrastiveImageSongModel(song_embedding_dim=song_embedding_dim, 
                                     embedding_dim=embedding_dim).to(device)
    model.load_state_dict(model_state)
    model.eval()
    
    # Run in selected mode
    if args.mode == 'build':
        if not args.song_data:
            raise ValueError("--song_data is required for build mode")
        
        build_song_database(
            model=model,
            song_data_path=args.song_data,
            save_path=args.database_path,
            device=device
        )
    
    elif args.mode == 'retrieve':
        if not args.image_path:
            raise ValueError("--image_path is required for retrieve mode")
        
        # Load song database
        if not os.path.exists(args.database_path):
            raise FileNotFoundError(f"Song database not found at {args.database_path}")
        
        song_database = torch.load(args.database_path)
        
        # Retrieve songs
        results = retrieve_songs_for_image(
            model=model,
            image_path=args.image_path,
            song_database=song_database,
            top_k=args.top_k,
            device=device
        )
        
        # Print results
        print(f"\nTop {args.top_k} songs for image {args.image_path}:")
        for i, song in enumerate(results):
            print(f"{i+1}. {song['title']} by {song['artist']} (similarity: {song['similarity']:.4f})")