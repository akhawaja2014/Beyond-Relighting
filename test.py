import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
from torchvision import transforms
from PIL import Image


# ----------- 1. Preprocessor Layer: HSH (27-ch) → RGB-like (3-ch) -----------
class HSHPreprocessor(nn.Module):
    def __init__(self, in_channels=27, out_channels=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.block(x)

# ----------- 2. Full Model: Preprocessor + Frozen ResNet-50 -----------
class HSHResNetPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.preprocessor = HSHPreprocessor(27, 3)
        self.resnet = models.resnet50(pretrained=True)
        self.resnet.fc = nn.Identity()  # Output = 2048-dim feature
        for param in self.resnet.parameters():
            param.requires_grad = False  # Freeze ResNet

    def forward(self, x):
        x = self.preprocessor(x)
        return self.resnet(x)  # (B, 2048)

# ----------- 3. Dummy Dataset: Replace with your actual HSH data loader -----------
class HSHDataset(Dataset):
    def __init__(self, root_dir, coeff_range=range(9), transform=None):
        """
        Args:
            root_dir (str): Root directory containing one subfolder per fragment (e.g., V1A, V1B).
            coeff_range (range): Which HSH coefficients to load (default: H0–H8).
            transform (callable, optional): Optional transform to apply to the stacked tensor.
        """
        self.root_dir = root_dir
        self.coeff_range = coeff_range
        self.transform = transform

        # Get list of subdirectories (fragment folders)
        self.fragment_dirs = sorted([
            os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        # Label map based on folder names (e.g., V1A, V1B → 0, 1, ...)
        self.label_map = {os.path.basename(d): i for i, d in enumerate(self.fragment_dirs)}

        self.base_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])

    def __len__(self):
        return len(self.fragment_dirs)

    def __getitem__(self, idx):
        fragment_path = self.fragment_dirs[idx]
        fragment_name = os.path.basename(fragment_path)

        # Stack HSH coefficient images: H0.jpg to H8.jpg
        coeff_tensors = []
        for i in self.coeff_range:
            coeff_file = os.path.join(fragment_path, f"plane_{i}.jpg")
            img = Image.open(coeff_file).convert('RGB')
            tensor = self.base_transform(img)  # shape (3, H, W)
            coeff_tensors.append(tensor)

        hsh_tensor = torch.cat(coeff_tensors, dim=0)  # shape: (27, H, W)

        # Optional normalization
        hsh_tensor = (hsh_tensor - hsh_tensor.mean()) / hsh_tensor.std()

        label = self.label_map[fragment_name]

        if self.transform:
            hsh_tensor = self.transform(hsh_tensor)

        return hsh_tensor, label

# ----------- 4. Training Loop -----------
def train(model, dataloader, device, epochs=50):
    # model.to(device)
    # model.train()
    # optimizer = torch.optim.Adam(model.preprocessor.parameters(), lr=1e-3)
    # classifier_head = nn.Linear(2048, 5).to(device)  # Add linear head for classification
    # criterion = nn.CrossEntropyLoss()

    model.to(device)
    model.train()

    num_classes = 2  # V and P
    classifier_head = nn.Linear(2048, num_classes).to(device)
    optimizer = torch.optim.Adam(model.preprocessor.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(epochs):
        total_loss = 0
        for hsh, label in dataloader:
            hsh, label = hsh.to(device), label.to(device)
            optimizer.zero_grad()

            features = model(hsh)
            logits = classifier_head(features)
            loss = criterion(logits, label)

            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Acc: {acc*100:.2f}%")

# ----------- 5. Run Training -----------
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dataset = HSHDataset('hsh_data')
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    model = HSHResNetPipeline()
    train(model, dataloader, device, epochs=100)
