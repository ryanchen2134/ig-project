import clip
import torch
from PIL import Image

# Load model
model, preprocess = clip.load("ViT-B/32", device="cpu")

# Load and preprocess image - This step resizes the images for the CLIP model

# image = preprocess(Image.open(r"C:\Users\yumic\OneDrive\Pictures\CLIP test\white_doggo_1.jpg")).unsqueeze(0)
# image = preprocess(Image.open(r"C:\Users\yumic\OneDrive\Pictures\CLIP test\happhy.jpg")).unsqueeze(0)
image = preprocess(Image.open(r"C:\Users\yumic\projects\ig_porject\ig-project\ulzzang\0ba9e110ece390196abc1a79f6090d5d.jpg")).unsqueeze(0)
# Tokenize text labels
texts = clip.tokenize(["Happy", "Sad", "Energetic", "Calm", "2010's"])

# Encode features
with torch.no_grad():
    image_features = model.encode_image(image)
    text_features = model.encode_text(texts)

    # Normalize features to unit length
    image_features /= image_features.norm(dim=-1, keepdim=True)
    text_features /= text_features.norm(dim=-1, keepdim=True)

    # Compute similarity (cosine similarity)
    similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

# Convert to numpy and print
probs = similarity.detach().cpu().numpy()
print("Label Probabilities:", probs)
