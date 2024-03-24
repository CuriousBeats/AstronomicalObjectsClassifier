import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
from PIL import Image
import os
from sklearn.model_selection import train_test_split
import shutil
from tqdm import tqdm
import time
batch_size = 32  # Adjust as needed
data_dir = "data"
processed_dir = data_dir + "/processed"
raw_dir = data_dir + "/raw/image_extracts/astroImages"
class_names = ['galaxy', 'qso', 'star'] 
for split in ['train', 'test', 'val']:
    for class_name in class_names:
        os.makedirs(os.path.join(processed_dir, split, class_name), exist_ok=True)

for class_name in class_names:
    source_dir = os.path.join(raw_dir, class_name)
    file_names = os.listdir(source_dir)

    # Split file names into train, test, and val sets
    train_files, test_files = train_test_split(
        file_names, test_size=0.2, random_state=42, stratify= [class_name] * len(file_names)
    )
    train_files, val_files = train_test_split(train_files, test_size=0.125, random_state=42, stratify= [class_name] * len(train_files))

    # Move files to respective directories 
    for file_name in train_files:
        shutil.move(os.path.join(source_dir, file_name), os.path.join(processed_dir, 'train', class_name, file_name))
    for file_name in test_files:
        shutil.move(os.path.join(source_dir, file_name), os.path.join(processed_dir, 'test', class_name, file_name))
    for file_name in val_files:
        shutil.move(os.path.join(source_dir, file_name), os.path.join(processed_dir, 'val', class_name, file_name))


data_transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),  # Converts PIL Image to PyTorch Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # ImageNet normalization
])

class ImageDataset(datasets.ImageFolder):
    def __init__(self, root, transform=None):
        super().__init__(root, transform=transform)

    def __getitem__(self, index):
        path, target = self.samples[index]
        image = Image.open(path).convert('RGB')  # Ensure RGB format

        if self.transform is not None:
            image = self.transform(image)

        return image, target

train_dataset = ImageDataset(root='data/processed/train', transform=data_transform)
test_dataset = ImageDataset(root='data/processed/test', transform=data_transform) 
val_dataset = ImageDataset(root='data/processed/val', transform=data_transform)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

class MyCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),  # Input channels = 3 for RGB
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),  # Flatten output of conv layers
            nn.Linear(64 * 32 * 32, 128),  # Adjust based on your image size
            nn.ReLU(),
            nn.Linear(128, 3)  # Output classes = 3
        )

    def forward(self, x):
        x = self.conv_layers(x)
        x = self.classifier(x)
        return x
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # Use GPU if available

model = MyCNN().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
epochs = 10  # Adjust number of epochs
# Create a torch log file to record training progress, using date and time
log_file = open(f'log_{time.strftime("%Y%m%d-%H%M%S")}.txt', 'w')
log_file.write(f'Epochs: {epochs}\n')
log_file.write(f'Batch size: {batch_size}\n')
log_file.write(f'Optimizer: Adam\n')
log_file.write(f'Learning rate: 0.001\n')
log_file.write(f'Loss function: CrossEntropyLoss\n')
log_file.write(f'Epoch: Step: Loss:\n')
#use tqdm to show progress bar
for epoch in tqdm(range(epochs)):
    for i, (images, labels) in enumerate(tqdm(train_loader)):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        if i % 100 == 0:  # Log progress every 100 mini-batches
            print(f'Epoch {epoch + 1}/{epochs}, Step {i + 1}, Loss: {loss.item():.4f}')
            log_file.write(f'{epoch + 1}/{epochs}, {i + 1}, {loss.item():.4f}\n')

correct = 0
total = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = correct / total
print(f'Accuracy on test set: {accuracy * 100:.2f}%')
