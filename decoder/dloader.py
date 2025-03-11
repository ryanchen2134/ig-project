from torch.utils.data import Dataset, DataLoader
import pandas as pd
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
        self.data = pd.read_csv(file_path)
        self.img_folder = img_folder
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path = f"{self.img_folder}/{self.data.iloc[index, 0]}"
        image = Image.open(img_path).convert('RGB')
        if(self.transform):
            image = self.transform(image)

        song_features = torch.tensor(self.data.iloc[index, 1:].values, dtype=torch.float32)
        return image, song_features