from torch.utils.data import Dataset, DataLoader
import pandas as pd
import os
import torch
import torchvision.transforms as transforms
from PIL import Image

#Based on PyTorch documentation
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom Dataset
class ImgSongDataset(Dataset):
    def __init__(self, file_path, img_folder):
        self.data = pd.read_csv(file_path, header = 0)
        print(f"Loaded DataFrame:\n{self.data.head()}")  # Check the first few rows
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_name = self.data.iloc[index]['image_filename']
        print(f"Loading Image: {img_name}")

        img_path = os.path.join(self.img_folder, img_name)
        #img_path = f"{self.img_folder}/{self.data.iloc[index, 0]}"
        image = Image.open(img_path).convert('RGB')
        if(self.transform):
            image = self.transform(image)

        #song_features = self.data.iloc[index]['song_feature_col1' : 'song_feature_colN'].values
        song_features = self.data.iloc[index, 1:]  # Assuming columns starting from 1 to the end are features
        song_features = pd.to_numeric(song_features, errors='coerce').fillna(0)  # Convert to numeric and replace NaN with 0
        song_features = torch.tensor(song_features.values, dtype=torch.float32)  # Convert to tensor
        return image, song_features