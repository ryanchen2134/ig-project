# inference/retrieval.py
import torch
import os
from PIL import Image
import torch.nn.functional as F
from typing import List, Dict, Any, Optional

from .database import SongDatabase

def retrieve_songs_for_image(model: torch.nn.Module, 
                            image_path: str, 
                            song_database: SongDatabase, 
                            top_k: int = 5, 
                            device: str = "cuda" if torch.cuda.is_available() else "cpu") -> List[Dict[str, Any]]:
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
    
    # Check if we're using CLIP or traditional preprocessing
    if hasattr(model.image_encoder, 'preprocess'):
        # CLIP-based model
        transform = model.image_encoder.preprocess
    else:
        # Traditional ResNet-based model
        from torchvision import transforms
        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Load and preprocess the image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # Get image embedding
    with torch.no_grad():
        image_embedding, _ = model(image_tensor, None, image_only=True)
    
    # Search for matching songs
    results = song_database.search_by_embedding(image_embedding.cpu(), top_k=top_k)
    
    return results

def get_song_for_shortcode(shortcode: str, song_database: SongDatabase) -> Optional[Dict[str, Any]]:
    """
    Get song data for a specific shortcode
    
    Args:
        shortcode: Image shortcode
        song_database: Database of songs
        
    Returns:
        Dictionary with song data or None if not found
    """
    return song_database.get_song_by_shortcode(shortcode)