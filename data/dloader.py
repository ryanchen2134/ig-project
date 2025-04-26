from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torch
import torchvision.transforms as transforms
from PIL import Image
import ast  

# Based on PyTorch documentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset
class ImgSongDataset(Dataset):
    def __init__(self, file_path, img_folder):
        self.data = pd.read_csv(file_path, header=0)
        print(f"Loaded DataFrame with {len(self.data)} rows")
        self.img_folder = img_folder
        self.transform = transform
        
        # Parse the audio_embedding strings to actual lists
        self.data['audio_embedding'] = self.data['audio_embedding'].apply(ast.literal_eval)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name = self.data.iloc[index]['shortcode']
        
        # Load and transform image
        img_path = os.path.join(self.img_folder, img_name + '.jpg')  # Adding file extension
        try:
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
        except FileNotFoundError:
            print(f"Warning: Image not found at {img_path}")
            # Create a blank image as fallback
            image = torch.zeros((3, 224, 224))
        
        # Get embedding and convert to tensor
        song_embeddings = torch.tensor(self.data.iloc[index]['audio_embedding'], dtype=torch.float32)
        
        return image, song_embeddings