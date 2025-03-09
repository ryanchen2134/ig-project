
import os
import torch
import clip
import pandas as pd
# from skimage import io, transform
# import numpy as np
# import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

#Ignore Warnings
# import warnings
# warnings.filterwarnings("ignore")
# plt.ion()

from PIL import Image
#Create Cystom Dataset Class
class ImageMoodDataset(Dataset):
    def __init__(self, file_path, root_dir, transform):
        self.data = pd.read_csv(file_path, skiprows=1)
        self.transform = transform
        self.root_dir = root_dir

        self.mood_to_idx = {mood: idx for idx, mood in enumerate(sorted(self.data.iloc[:, 1].unique()))}


    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        if torch.is_tensor(index):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir, self.data.iloc[index, 0])
        mood_label = self.data.iloc[index, 1]
        
        image = Image.open(img_name).convert('RGB')
        image = self.transform(image)

        mood_idx = self.mood_to_idx[mood_label]
        mood_tensor = torch.tensor(mood_idx, dtype=torch.long)

        return image, mood_tensor, img_name
    