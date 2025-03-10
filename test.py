import clip
import torch
import data_loader
import numpy as np
import torch.multiprocessing as mp

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from sentence_transformers import SentenceTransformer
from data_loader import ImageMoodDataset

#Trying torch multiprocessing
try:
    mp.set_start_method('spawn', force= True)
except RuntimeError:
    pass

# Load and preprocess image - This step resizes the images for the CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
#Want to use ViT-L/32s
model, preprocess = clip.load("ViT-B/32", device=device)

if __name__ == "__main__":
    model.eval()

    r_dir = 'G:/My Drive/YS - Images/City pop aesthetic'
    f_path = './filtered/new_city_pop.csv'

    dataset = ImageMoodDataset(file_path=f_path, root_dir=r_dir, transform=preprocess)
    dataloader = DataLoader(dataset=dataset, batch_size=32, shuffle=True, num_workers=4)    
    
    #Tokenize
    TOKENS = ["happy", "sad", "energetic", "calm", "2010's", "beauty", "female", "male", "japanese city pop", "city pop", "80's"]
    texts = clip.tokenize(TOKENS).to(device)

    # Encode features
    with torch.no_grad():
        for batch_idx, (images, moods, file_paths) in enumerate(dataloader):
            images = images.to(device)
            
            #Batch Number Message
            print(f"\nProcessing Batch {batch_idx + 1}...")

            image_features = model.encode_image(images)
            text_features = model.encode_text(texts)

            # Normalize features to unit length
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)

            # Compute similarity (cosine similarity)
            similarity = (100.0 * image_features @ text_features.T).softmax(dim=-1)

            # Convert to numpy and print
            probs = similarity.detach().cpu().numpy()
            probs = np.round(probs, 2)

            for img_idx, prob in enumerate(probs):
                file_path = file_path[img_idx]
                print(f"Image {img_idx +1}: {file_path}")
                for token, p in zip(TOKENS, prob):
                    print(f"Label: {token} [{float(p) * 100:.2f}%]")
