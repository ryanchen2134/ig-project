import torch
import os
from PIL import Image
import matplotlib.pyplot as plt
import argparse
from models.decoder.contrastive import ContrastiveImageSongModel
from retrieval import prepare_song_database, build_song_database, retrieve_songs_for_image

def test_with_sample_data(model_path, original_csv_path, test_image_path, device='cuda'):
    """
    Test the retrieval pipeline with sample data
    
    Args:
        model_path: Path to the trained model
        original_csv_path: Path to original CSV with shortcode, link, embedding
        test_image_path: Path to test image
        device: Device to use for processing
    """
    print("=== TESTING RETRIEVAL PIPELINE ===")
    
    # Load model
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=device)
    
    # Extract model parameters from checkpoint
    if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
        model_state = checkpoint['model_state_dict']
        embedding_dim = checkpoint.get('embedding_dim', 64)
        backbone_type = checkpoint.get('backbone_type', 'resnet18')
        song_embedding_dim = checkpoint.get('song_embedding_dim', 6144)  # Default if not specified
        
        print(f"Model info - Backbone: {backbone_type}, Embedding dim: {embedding_dim}, Song embedding dim: {song_embedding_dim}")
    else:
        print("Warning: Checkpoint format not recognized, using default parameters")
        model_state = checkpoint
        embedding_dim = 64
        backbone_type = 'resnet18'
        song_embedding_dim = 6144
    
    # Initialize model with the correct parameters
    model = ContrastiveImageSongModel(
        song_embedding_dim=song_embedding_dim,
        embedding_dim=embedding_dim,
        backbone_type=backbone_type
    ).to(device)
    
    model.load_state_dict(model_state)
    model.eval()
    print("Model loaded successfully.")
    
    # First prepare song data from the original CSV
    prepared_csv_path = "prepared_song_data.csv"
    print(f"Preparing song database from {original_csv_path}...")
    
    try:
        song_df = prepare_song_database(original_csv_path, prepared_csv_path)
        print(f"Prepared song data saved to {prepared_csv_path}")
        print(f"Found {len(song_df)} unique songs in the dataset")
    except Exception as e:
        print(f"Error preparing song database: {e}")
        return
    
    # Build the test database with the prepared data
    test_db_path = "test_song_database.pt"
    print(f"Building song database with model projections...")
    
    try:
        song_db = build_song_database(
            model=model,
            song_data_path=prepared_csv_path,
            save_path=test_db_path,
            device=device
        )
        print("Song database built successfully.")
    except Exception as e:
        print(f"Error building song database: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # Test retrieval with a sample image
    print(f"\nTesting retrieval with image: {test_image_path}")
    
    try:
        # Load and display the test image
        img = Image.open(test_image_path).convert('RGB')
        plt.figure(figsize=(6, 6))
        plt.imshow(img)
        plt.title("Query Image")
        plt.axis('off')
        plt.show()
        
        # Retrieve matching songs
        results = retrieve_songs_for_image(
            model=model,
            image_path=test_image_path,
            song_database=song_db,
            top_k=5,  # Show top 5 matches
            device=device
        )
        
        # Print results
        print("\nTop matching songs:")
        for i, song in enumerate(results):
            print(f"{i+1}. {song['title']} by {song['artist']} (similarity: {song['similarity']:.4f})")
            
        print("\nRetrieval test completed successfully!")
        
    except Exception as e:
        print(f"Error during retrieval: {e}")
        import traceback
        traceback.print_exc()
    
    # Clean up test files if needed
    if os.path.exists(test_db_path) and input("\nRemove test database? (y/n): ").lower() == 'y':
        os.remove(test_db_path)
        print("Test database removed.")
    
    if os.path.exists(prepared_csv_path) and input("\nRemove prepared song data CSV? (y/n): ").lower() == 'y':
        os.remove(prepared_csv_path)
        print("Prepared song data CSV removed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Test retrieval pipeline')
    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to the trained model checkpoint (e.g., contrastive_model_resnet18.pth)')
    parser.add_argument('--original_csv', type=str, required=True,
                       help='Path to original CSV with song data including audio embeddings')
    parser.add_argument('--test_image', type=str, required=True,
                       help='Path to test image for retrieval')
    
    args = parser.parse_args()
    
    test_with_sample_data(
        model_path=args.model_path,
        original_csv_path=args.original_csv,
        test_image_path=args.test_image
    )