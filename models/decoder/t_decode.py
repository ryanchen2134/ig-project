import torch
import pandas as pd
from models.decoder.img_decoder import ImageEncoder
from models.decoder.song_decode import SongEncoder
from torchvision import transforms
import torch.nn as nn
from PIL import Image
from models.decoder.contrastive import ContrastiveImageSongModel
import matplotlib.pyplot as plt

#Image testing
def test_image_encoder_output_shape():
    # Initialize the ImageEncoder
    embedding_dim = 128
    encoder = ImageEncoder(embedding_dim=embedding_dim)
    encoder.eval()  # Set to evaluation mode

    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    batch_size = 1
    
    #dummy_input = torch.randn(batch_size, 3, 224, 224)  # Example: batch size of 4, 3 color channels, 224x224 image
    img_path = "models/decoder/test_img/happhy.jpg"
    image = Image.open(img_path).convert("RGB")
    input_batch = preprocess(image).unsqueeze(0)  # Add batch dimension
    
    if torch.cuda.is_available():
        input_batch = input_batch.to("cuda")
        encoder.to("cuda")

    with torch.no_grad():
        embedding = encoder(input_batch)

    print(f"Embedding shape: {embedding.shape}")
    print(f"First few values: {embedding[0, :10]}")
    print(f"Embedding norm: {torch.norm(embedding).item()}")

    # Pass the dummy input through the encoder
    # output = encoder(dummy_input)
    
    # Assert the output shape is correct
    assert embedding.shape == (batch_size, embedding_dim), f"Expected shape {(batch_size, embedding_dim)}, but got {output.shape}"

def test_image_encoder_normalization():
    # Initialize the ImageEncoder
    encoder = ImageEncoder(embedding_dim=128)
    encoder.eval()  # Set to evaluation mode
    
    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 224, 224)  # Single image
    
    # Pass the dummy input through the encoder
    output = encoder(dummy_input)
    
    # Compute the norm of the output embeddings
    norm = torch.norm(output, p=2, dim=1)
    
    # Assert that the norm is approximately 1 (normalized embeddings)
    assert torch.allclose(norm, torch.tensor(1.0), atol=1e-6), f"Expected norm 1.0, but got {norm}"


# Song testing
def test_song_encoder_output_shape():
    df = pd.read_csv("DISCO/song_csv/Lady Gaga-Die with a smile.csv", header=None)
    #print(df.head())  # See what was actually loaded
    #print(df.shape)   # Check if rows exist
    vec = df.iloc[0].tolist()
    vec = torch.tensor(vec, dtype=torch.float32)
    vec = vec.unsqueeze(0)  # Add batch dimension
    # Initialize the ImageEncoder
    embedding_dim = 128
#    encoder = SongEncoder(embedding_dim=embedding_dim)
    encoder = SongEncoder(input_dim=58, embedding_dim=embedding_dim)
    encoder.eval()  # Set to evaluation mode
    # Create a dummy input tensor with shape (batch_size, channels, height, width)
    batch_size = 1
    dummy_input = torch.randn(batch_size, 3, 224, 224)  # Example: batch size of 4, 3 color channels, 224x224 image
    
    # Pass the dummy input through the encoder
    output = encoder(vec)
    
    # Assert the output shape is correct
    assert output.shape == (batch_size, embedding_dim), f"Expected shape {(batch_size, embedding_dim)}, but got {output.shape}"

def test_song_encoder_normalization():
    # Initialize the ImageEncoder
    encoder = SongEncoder(embedding_dim=128)
    encoder.eval()  # Set to evaluation mode
    
    # Create a dummy input tensor
    dummy_input = torch.randn(1, 3, 224, 224)  # Single image
    
    # Pass the dummy input through the encoder
    output = encoder(dummy_input)
    
    # Compute the norm of the output embeddings
    norm = torch.norm(output, p=2, dim=1)
    
    # Assert that the norm is approximately 1 (normalized embeddings)
    assert torch.allclose(norm, torch.tensor(1.0), atol=1e-6), f"Expected norm 1.0, but got {norm}"

    def test_contrastive_model():

        # Initialize the ContrastiveModel
        embedding_dim = 128
        model = ContrastiveModel(image_embedding_dim=embedding_dim, song_embedding_dim=embedding_dim)
        model.eval()  # Set to evaluation mode

        # Create dummy inputs for image and song embeddings
        batch_size = 4
        image_embeddings = torch.randn(batch_size, embedding_dim)
        song_embeddings = torch.randn(batch_size, embedding_dim)

        # Pass the embeddings through the contrastive model
        with torch.no_grad():
            similarity_scores = model(image_embeddings, song_embeddings)

        # Assert the output shape is correct
        assert similarity_scores.shape == (batch_size,), f"Expected shape {(batch_size,)}, but got {similarity_scores.shape}"

        # Check if similarity scores are within a valid range (e.g., -1 to 1 for cosine similarity)
        assert torch.all(similarity_scores >= -1.0) and torch.all(similarity_scores <= 1.0), \
            f"Similarity scores out of range: {similarity_scores}"

        print(f"Similarity scores: {similarity_scores}")
        print("Contrastive model test passed!")

def test_contrastive_model_with_paired_sample(model_paths, image_paths, song_feature_list, song_feature_dim, embedding_dim = 128, device = 'cuda'):
    # Initialize the ContrastiveModel
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    embedding_dim = 128
    model = ContrastiveModel(image_embedding_dim=embedding_dim, song_embedding_dim=embedding_dim)
    model.load_state.dict(torch.load(model_path))
    model = model.to(device)
    model.eval()  # Set to evaluation mode

    # Create paired samples for image and song embeddings
    batch_size = 4

    image_embeddings = []
    for img_path in img_paths:
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.get_image_embeddings(image_tensor)
            image_embeddings.append(embedding)

    song_embeddings = []
    for song_path in song_feature_list:
        song_tensor = torch.tensor(song_feat, dtype = torch.float32).unsqueeze(0).to(device)

        with torch.no_grad():
            embedding = model.get_song_embedding(song_tensor)
            song_embeddings.append(embedding)
        
    image_embeddings = torch.cat(image_embeddings, dim=0)
    song_embeddings = torch.cat(song_embeddings, dim=0)
    
    similarity_matrix = torch.nn.functional.cosine_similarity(
        image_embedding.unsqueeze(1), song_embeddings.unqsueeze(0), dim = 2
    )

    # Metrics
    paired_similarities = []
    ranks = []
    top1_accuracy = 0
    top5_accuracy = 0
    
    for i in range(len(image_paths)):
        # Get similarities for this image with all songs
        sims = similarity_matrix[i].cpu().numpy()
        
        # Get similarity with paired song
        paired_sim = sims[i]
        paired_similarities.append(paired_sim)
        
        # Get rank of paired song (ascending order, so need to sort negated similarities)
        rank = np.where(np.argsort(-sims) == i)[0][0] + 1  # +1 because ranks start at 1
        ranks.append(rank)
        
        # Update top-k accuracy
        if rank == 1:
            top1_accuracy += 1
        if rank <= 5:
            top5_accuracy += 1
    
    # Calculate final metrics
    top1_accuracy /= len(image_paths)
    top5_accuracy /= len(image_paths)
    mean_rank = np.mean(ranks)
    median_rank = np.median(ranks)
    mean_reciprocal_rank = np.mean([1/r for r in ranks])
    mean_paired_similarity = np.mean(paired_similarities)
    
    # Return metrics
    metrics = {
        'top1_accuracy': top1_accuracy,
        'top5_accuracy': top5_accuracy,
        'mean_rank': mean_rank,
        'median_rank': median_rank,
        'mean_reciprocal_rank': mean_reciprocal_rank,
        'mean_paired_similarity': mean_paired_similarity,
        'paired_similarities': paired_similarities,
        'ranks': ranks
    }
    
    return metrics, similarity_matrix.cpu().numpy()

def test_func():
    # Initialize a new model with random weights
    
    model = ContrastiveImageSongModel(song_feature_dim=song_feature_dim, embedding_dim=embedding_dim)

    # Instead of loading weights, just use the randomly initialized model
    # model.load_state_dict(torch.load(model_path))  # Skip this line

    # Test the untrained model
    similarity = test_paired_sample(
        model=model,  # Pass the model directly instead of the path
        image_path=image_path,
        song_features=song_features,
        song_feature_dim=song_feature_dim
    )
    


if __name__ == "__main__":
    #test_image_encoder_output_shape()
    #test_image_encoder_normalization()
    #test_song_encoder_output_shape()
    #test_song_encoder_normalization()

    # NEEDS model_paths, image_paths, song_feature_list, song_feature_dim
    test_func()
    print("All tests passed!")