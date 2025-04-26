import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
import torch.optim as optim
from data.dloader import ImgSongDataset
from torch.utils.data import DataLoader
from PIL import Image

# Load Pretrained ResNet model
model = models.resnet50(weights="IMAGENET1K_V2")
model = torch.nn.Sequential(*list(model.children())[:-1])  # Remove classification layer

num_song_features = 10
model.fc = nn.Linear(2048, num_song_features)

model.train()

# Image Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def extract_features(image_path):
    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0)  # Add batch dimension
    with torch.no_grad():
        features = model(image)
    return features.squeeze().numpy()

# Example usage
image_features = extract_features(r"G:\My Drive\YS - Images\aquarium aesthetic\0a36a54dbcb608e26c8b2e6e24dfdf5a.jpg")
print(image_features.shape)  # Shape will be (2048,)

dataset = ImgSongDataset(file_path ="", img_folder ="", transform = transform)
dataloader = DataLoader(dataset, batch_size= 32, shuffle=True)


#Training
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

num_epochs = 10
for epoch in range(num_epochs):
    running_loss = 0.0
    for images, song_features in dataloader:
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, song_features)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Epoch {epoch +1}, Loss: {running_loss/len(dataloader)}")
    
print("Training Complete!")
    