import torch
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import ast
import numpy as np
from typing import Tuple, Callable, Optional, Union

class ImgSongDataset(Dataset):
    """Dataset for loading image-song pairs"""
    
    def __init__(self, file_path: str, 
                 img_folder: str, 
                 transform: Optional[Callable] = None,
                 image_column: str = 'image_path',
                 audio_embedding_column: str = 'audio_embedding'):
        """
        Initialize the dataset
        
        Args:
            file_path: Path to CSV file with image paths and song features
            img_folder: Path to folder containing images
            transform: Transforms to apply to images
            image_column: Column name for image paths
            audio_embedding_column: Column name for audio embeddings
        """
        self.data = pd.read_csv(file_path)
        self.img_folder = img_folder
        self.transform = transform
        self.image_column = image_column
        self.audio_embedding_column = audio_embedding_column
        
        # Convert string embeddings to arrays if needed
        if isinstance(self.data[audio_embedding_column].iloc[0], str):
            self.data[audio_embedding_column] = self.data[audio_embedding_column].apply(
                lambda x: np.array(ast.literal_eval(x), dtype=np.float32))
            
    def __len__(self) -> int:
        return len(self.data)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get an image-song pair
        
        Args:
            idx: Index of the pair
            
        Returns:
            Tuple of (image_tensor, song_features_tensor)
        """
        row = self.data.iloc[idx]
        
        # Get image path and load image
        img_path = os.path.join(self.img_folder, row[self.image_column])
        image = Image.open(img_path).convert('RGB')
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        
        # Get song features
        song_features = torch.tensor(row[self.audio_embedding_column], dtype=torch.float32)
        
        return image, song_features

    @staticmethod
    def get_clip_preprocess():
        """Get CLIP preprocessing transform"""
        import clip
        _, preprocess = clip.load("ViT-B/32", device="cpu")
        return preprocess