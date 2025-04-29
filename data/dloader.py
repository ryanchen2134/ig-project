from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import os
import torch
import torchvision.transforms as transforms
from PIL import Image, ImageFilter
import ast  

# Based on PyTorch documentation

# transform = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
# ])

# Stronger Transformation for a smaller dataset
transform = transforms.Compose([
    transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)], p=0.8),
    transforms.RandomGrayscale(p=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Gaussian blur augmentation
class GaussianBlur:
    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma
        
    def __call__(self, x):
        from PIL import ImageFilter
        sigma = np.random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x
    
# Custom Dataset
class ImgSongDataset(Dataset):
    def __init__(self, file_path, img_folder, transform=None, num_views=2, is_train=True):
        self.data = pd.read_csv(file_path, header=0)
        print(f"Loaded DataFrame with {len(self.data)} rows")
        self.img_folder = img_folder
        self.num_views = num_views if is_train else 1
        self.is_train = is_train
        
        # Parse the audio_embedding strings to actual lists if they're strings
        if isinstance(self.data['audio_embedding'].iloc[0], str):
            import ast
            self.data['audio_embedding'] = self.data['audio_embedding'].apply(ast.literal_eval)
        

        
        # Strong augmentation for contrastive learning (SimCLR-style)
        if transform is None and is_train:
            self.transform = transforms.Compose([
                transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([
                    transforms.ColorJitter(0.4, 0.4, 0.4, 0.2)
                ], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([GaussianBlur([.1, 2.])], p=0.5),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
            ])
        elif transform is not None:
            self.transform = transform
        else:
            # Evaluation transform
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                                    std=[0.229, 0.224, 0.225])
            ])
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        
        # Get image path and audio embeddings
        img_name = self.data.iloc[idx]['shortcode']
        img_path = os.path.join(self.img_folder, img_name + '.jpg')  # Adding file extension
        audio_embeddings = self.data.iloc[idx]['audio_embedding']
        
        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            print(f"Warning: Image not found at {img_path}")
            # Create a blank image as fallback
            image = Image.new('RGB', (224, 224), color='gray')
            
        # Apply transforms and generate multiple views if training
        if self.is_train and self.num_views > 1:
            views = [self.transform(image) for _ in range(self.num_views)]
            return views, torch.tensor(audio_embeddings, dtype=torch.float32)
        else:
            # Single view (for validation/testing)
            image = self.transform(image)
            return image, torch.tensor(audio_embeddings, dtype=torch.float32)