# inference/database.py

import os
import torch
import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Union, Any
import logging

logger = logging.getLogger(__name__)

class SongDatabase:
    """
    Manage a database of song embeddings and metadata for retrieval
    """
    def __init__(self, database_path: str = "song_database.pt"):
        """
        Initialize the song database
        
        Args:
            database_path: Path to save/load the database
        """
        self.database_path = database_path
        
        # Initialize empty database structure
        self.database = {
            'embeddings': None,           # Tensor of song embeddings
            'shortcodes': [],             # List of shortcodes (matching image filenames)
            'titles': [],                 # List of song titles
            'artists': [],                # List of artist names
            'links': [],                  # List of image links
            'metadata': {},               # Dictionary of additional metadata
            'shortcode_to_idx': {}        # Mapping from shortcode to index
        }
        
        # Load existing database if available
        if os.path.exists(database_path):
            self.load()
    
    def load(self) -> None:
        """Load database from file"""
        try:
            self.database = torch.load(self.database_path)
            logger.info(f"Loaded database with {len(self.database['shortcodes'])} songs from {self.database_path}")
            
            # Build index mapping if not present in older versions
            if 'shortcode_to_idx' not in self.database:
                self._rebuild_index()
                
        except Exception as e:
            logger.error(f"Error loading database: {e}")
            # Keep the initialized empty database
    
    def save(self) -> None:
        """Save database to file"""
        try:
            torch.save(self.database, self.database_path)
            logger.info(f"Saved database with {len(self.database['shortcodes'])} songs to {self.database_path}")
        except Exception as e:
            logger.error(f"Error saving database: {e}")
    
    def _rebuild_index(self) -> None:
        """Rebuild the shortcode to index mapping"""
        self.database['shortcode_to_idx'] = {
            shortcode: idx for idx, shortcode in enumerate(self.database['shortcodes'])
        }
    
    def build_from_csv(self, csv_path: str, embedding_column: str = 'embedding') -> None:
        """
        Build database from a CSV file with song data and embeddings
        
        Args:
            csv_path: Path to CSV file
            embedding_column: Column containing embeddings
        """
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            
            required_columns = ['shortcode', 'music_title', 'music_artist']
            if not all(col in df.columns for col in required_columns):
                raise ValueError(f"CSV must contain columns: {required_columns}")
            
            # Extract data
            shortcodes = df['shortcode'].tolist()
            titles = df['music_title'].tolist()
            artists = df['music_artist'].tolist()
            
            # Get links if available
            links = df['link'].tolist() if 'link' in df.columns else [''] * len(shortcodes)
            
            # Process embeddings
            embeddings_list = []
            for _, row in df.iterrows():
                if embedding_column in row:
                    # Handle different embedding formats
                    embedding = row[embedding_column]
                    if isinstance(embedding, str):
                        # Convert string representation to list
                        try:
                            import ast
                            embedding = ast.literal_eval(embedding)
                        except (ValueError, SyntaxError):
                            logger.error(f"Could not parse embedding for {row['shortcode']}")
                            continue
                    
                    embeddings_list.append(embedding)
                else:
                    logger.error(f"Embedding column '{embedding_column}' not found in row")
                    continue
            
            # Convert to tensor
            embeddings = torch.tensor(embeddings_list, dtype=torch.float32)
            
            # Update database
            self.database['embeddings'] = embeddings
            self.database['shortcodes'] = shortcodes
            self.database['titles'] = titles
            self.database['artists'] = artists
            self.database['links'] = links
            
            # Build index mapping
            self._rebuild_index()
            
            # Save updated database
            self.save()
            
            logger.info(f"Built database with {len(shortcodes)} songs from {csv_path}")
            
        except Exception as e:
            logger.error(f"Error building database from CSV: {e}")
            raise
    
    def add_song(self, 
                shortcode: str, 
                title: str, 
                artist: str, 
                embedding: torch.Tensor, 
                link: str = "",
                additional_metadata: Dict[str, Any] = None) -> None:
        """
        Add a single song to the database
        
        Args:
            shortcode: Shortcode (matches image filename)
            title: Song title
            artist: Artist name
            embedding: Song embedding tensor
            link: Optional image link
            additional_metadata: Optional additional metadata
        """
        # Ensure embedding is a 1D tensor
        if len(embedding.shape) > 1:
            embedding = embedding.squeeze()
        
        # Check if song already exists
        if shortcode in self.database['shortcode_to_idx']:
            # Update existing entry
            idx = self.database['shortcode_to_idx'][shortcode]
            self.database['titles'][idx] = title
            self.database['artists'][idx] = artist
            self.database['links'][idx] = link
            
            # Update embedding
            if self.database['embeddings'] is None:
                self.database['embeddings'] = embedding.unsqueeze(0)
            else:
                self.database['embeddings'][idx] = embedding
                
            # Update metadata
            if additional_metadata:
                if shortcode not in self.database['metadata']:
                    self.database['metadata'][shortcode] = {}
                self.database['metadata'][shortcode].update(additional_metadata)
                
            logger.info(f"Updated song in database: {artist} - {title}")
        else:
            # Add new entry
            self.database['shortcodes'].append(shortcode)
            self.database['titles'].append(title)
            self.database['artists'].append(artist)
            self.database['links'].append(link)
            
            # Update shortcode to index mapping
            new_idx = len(self.database['shortcodes']) - 1
            self.database['shortcode_to_idx'][shortcode] = new_idx
            
            # Add embedding
            if self.database['embeddings'] is None:
                self.database['embeddings'] = embedding.unsqueeze(0)
            else:
                self.database['embeddings'] = torch.cat([
                    self.database['embeddings'],
                    embedding.unsqueeze(0)
                ], dim=0)
                
            # Add metadata
            if additional_metadata:
                self.database['metadata'][shortcode] = additional_metadata
                
            logger.info(f"Added new song to database: {artist} - {title}")
        
        # Save database after each addition
        self.save()
    
    def remove_song(self, shortcode: str) -> bool:
        """
        Remove a song from the database
        
        Args:
            shortcode: Shortcode to remove
            
        Returns:
            True if removed, False if not found
        """
        if shortcode not in self.database['shortcode_to_idx']:
            return False
        
        # Get index
        idx = self.database['shortcode_to_idx'][shortcode]
        
        # Remove from lists
        del self.database['shortcodes'][idx]
        del self.database['titles'][idx]
        del self.database['artists'][idx]
        del self.database['links'][idx]
        
        # Remove from embeddings tensor
        if self.database['embeddings'] is not None:
            indices = torch.ones(len(self.database['embeddings']), dtype=torch.bool)
            indices[idx] = False
            self.database['embeddings'] = self.database['embeddings'][indices]
        
        # Remove from metadata
        if shortcode in self.database['metadata']:
            del self.database['metadata'][shortcode]
        
        # Rebuild index mapping
        self._rebuild_index()
        
        # Save updated database
        self.save()
        
        return True
    
    def get_song_by_shortcode(self, shortcode: str) -> Optional[Dict[str, Any]]:
        """
        Get song data by shortcode
        
        Args:
            shortcode: Shortcode to lookup
            
        Returns:
            Dictionary with song data or None if not found
        """
        if shortcode not in self.database['shortcode_to_idx']:
            return None
        
        idx = self.database['shortcode_to_idx'][shortcode]
        
        return {
            'shortcode': shortcode,
            'title': self.database['titles'][idx],
            'artist': self.database['artists'][idx],
            'link': self.database['links'][idx],
            'embedding': self.database['embeddings'][idx] if self.database['embeddings'] is not None else None,
            'metadata': self.database['metadata'].get(shortcode, {})
        }
    
    def search_by_embedding(self, query_embedding: torch.Tensor, top_k: int = 5) -> List[Dict[str, Any]]:
        """
        Search for songs by embedding similarity
        
        Args:
            query_embedding: Query embedding
            top_k: Number of results to return
            
        Returns:
            List of dictionaries with song data
        """
        if self.database['embeddings'] is None or len(self.database['shortcodes']) == 0:
            return []
        
        # Ensure query is a tensor
        if not isinstance(query_embedding, torch.Tensor):
            query_embedding = torch.tensor(query_embedding, dtype=torch.float32)
        
        # Normalize query
        query_embedding = query_embedding / query_embedding.norm()
        
        # Compute similarities
        similarities = torch.matmul(self.database['embeddings'], query_embedding.squeeze())
        
        # Get top-k indices
        top_indices = torch.argsort(similarities, descending=True)[:top_k]
        
        # Collect results
        results = []
        for idx in top_indices:
            idx = idx.item()
            shortcode = self.database['shortcodes'][idx]
            
            results.append({
                'shortcode': shortcode,
                'title': self.database['titles'][idx],
                'artist': self.database['artists'][idx],
                'link': self.database['links'][idx],
                'similarity': similarities[idx].item(),
                'metadata': self.database['metadata'].get(shortcode, {})
            })
        
        return results
    
    def get_all_songs(self) -> List[Dict[str, Any]]:
        """
        Get all songs in the database
        
        Returns:
            List of dictionaries with song data
        """
        return [
            {
                'shortcode': self.database['shortcodes'][i],
                'title': self.database['titles'][i],
                'artist': self.database['artists'][i],
                'link': self.database['links'][i],
                'metadata': self.database['metadata'].get(self.database['shortcodes'][i], {})
            }
            for i in range(len(self.database['shortcodes']))
        ]
    
    def __len__(self) -> int:
        """Get number of songs in the database"""
        return len(self.database['shortcodes'])