
import os
from torch.utils.tensorboard import SummaryWriter
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from torch.utils.data import random_split
from PIL import Image
from collections import Counter
#import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import Subset





# ==============================
# 1. Dataset Class (V vs P)
# ==============================
class FolderHSHDataset(Dataset):
    def __init__(self, root_dir, coeff_range=range(9), image_size=224):
        self.root_dir = root_dir
        self.coeff_range = coeff_range
        self.image_size = image_size

        self.fragment_dirs = sorted([
            os.path.join(root_dir, d) for d in os.listdir(root_dir)
            if os.path.isdir(os.path.join(root_dir, d))
        ])

        self.base_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.fragment_dirs)

    def __getitem__(self, idx):
        fragment_path = self.fragment_dirs[idx]
        fragment_name = os.path.basename(fragment_path)

        coeff_tensors = []
        for i in self.coeff_range:
            coeff_file = os.path.join(fragment_path, f"plane_{i}.jpg")
            img = Image.open(coeff_file).convert('RGB')
            tensor = self.base_transform(img)
            coeff_tensors.append(tensor)

        hsh_tensor = torch.cat(coeff_tensors, dim=0)  # Shape: [27, H, W]
        hsh_tensor = (hsh_tensor - hsh_tensor.mean()) / hsh_tensor.std()

        # Label: 0 for V-series, 1 for P-series
        if fragment_name.startswith('V'):
            label = 0
        elif fragment_name.startswith('P'):
            label = 1
        else:
            raise ValueError(f"Unrecognized class in {fragment_name}")

        return hsh_tensor, label

# ==============================
# 2. Preprocessing Layer + Frozen ResNet
# ==============================
class HSHPreprocessor(nn.Module):
    def __init__(self, in_channels=27, out_channels=3):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(32, 16, kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(16, out_channels, kernel_size=1)
        )

    def forward(self, x):
        return self.block(x)

class HSHResNetPipeline(nn.Module):
    def __init__(self):
        super().__init__()
        self.preprocessor = HSHPreprocessor(27, 3)
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        self.resnet.fc = nn.Identity()  # Output = 2048-dim features
        for param in self.resnet.parameters():
            param.requires_grad = False

    def forward(self, x):
        x = self.preprocessor(x)
        return self.resnet(x)  # (B, 2048)

# ==============================
# 3. Training Loop
# ==============================
def train(model, classifier_head, train_loader, val_loader, device, optimizer, criterion, epochs):
    model.to(device)
    model.train()

    num_classes = 2  # V and P
    # classifier_head = nn.Linear(2048, num_classes).to(device)
    # optimizer = torch.optim.Adam(model.preprocessor.parameters(), lr=1e-3)
    # criterion = nn.CrossEntropyLoss()

    train_losses, val_losses = [], []
    train_accuracies, val_accuracies = [], []

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for hsh, label in train_loader:
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

        train_acc = correct / total
        #print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Acc: {acc*100:.2f}%")
        avg_train_loss = total_loss / len(train_loader)
        #train_losses.append(avg_train_loss)
        #train_accuracies.append(train_acc)

        #val_loss, val_acc = evaluate(model, val_loader, classifier_head, device, criterion)
        val_loss, val_acc = evaluate(model, val_loader, classifier_head, device, criterion)
        #val_losses.append(val_loss)
        #val_accuracies.append(val_acc)

        writer.add_scalar("Loss/Train", avg_train_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Loss/Val", val_loss, epoch)
        writer.add_scalar("Accuracy/Val", val_acc, epoch)


        print(f"Epoch {epoch+1}/{epochs} - Loss: {total_loss:.4f} - Train Acc: {train_acc*100:.2f}% - Val Acc: {val_acc*100:.2f}% - Val Loss: {val_loss:.4f}")
    
    #return train_losses, val_losses, train_accuracies, val_accuracies

# ==============================
# 4. Run Training
# ==============================

def evaluate(model, dataloader, classifier_head, device, criterion):
    model.eval()
    classifier_head.eval()
    total_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for hsh, label in dataloader:
            hsh, label = hsh.to(device), label.to(device)
            features = model(hsh)
            logits = classifier_head(features)
            loss = criterion(logits, label)
            total_loss += loss.item()
            pred = torch.argmax(logits, dim=1)
            correct += (pred == label).sum().item()
            total += label.size(0)
    avg_loss = total_loss / len(dataloader)
    acc = correct / total
    return avg_loss, acc


if __name__ == "__main__":
    # Set seeds for reproducibility
    seed = 42
    # torch.manual_seed(seed)
    # torch.cuda.manual_seed(seed)
    # torch.cuda.manual_seed_all(seed)

    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    np.random.seed(seed)
    

    log_dir = f"runs/fragment_matching_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    print(f"Logging to {log_dir}")
    
    # writer = SummaryWriter(log_dir="runs/fragment_matching")

    # # Initialize W&B
    # wandb.init(mode="offline",project="fragment-matching", config={
    #     "epochs": 5,
    #     "batch_size": 8,
    #     "learning_rate": 1e-3,
    #     "architecture": "HSHResNetPipeline",
    #     "dataset": "FolderHSHDataset",
    #     "split_ratio": [0.7, 0.15, 0.15],
    #     "seed": seed
    # })

    root = "E:/PHD/Project/Papers/Journal_Oseberg/fragmentmatchingcode/hsh_data/"
    dataset = FolderHSHDataset(root_dir=root)

    # Data splits
    total_len = len(dataset)
    train_len = int(0.7 * total_len)
    val_len = int(0.15 * total_len)
    test_len = total_len - train_len - val_len

    generator = torch.Generator().manual_seed(seed)

    # to be replaced
    #train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=generator)
    # Step 1: Extract labels from the dataset
    all_labels = [dataset[i][1] for i in range(len(dataset))]
    all_indices = list(range(len(dataset)))

    # Step 2: First split -> train + temp
    train_indices, temp_indices, train_labels, temp_labels = train_test_split(
        all_indices,
        all_labels,
        test_size=0.3,  # 30% for val+test
        stratify=all_labels,
        random_state=seed
    )

    # Step 3: Second split -> val + test
    val_indices, test_indices = train_test_split(
        temp_indices,
        test_size=0.5,  # 50% of temp = 15% of total
        stratify=temp_labels,
        random_state=seed
    )

    # Step 4: Wrap with Subset
    train_set = Subset(dataset, train_indices)
    val_set = Subset(dataset, val_indices)
    test_set = Subset(dataset, test_indices)

    # val_labels = [dataset[idx][1] for idx in val_set.indices]
    # print("Validation label distribution:", Counter(val_labels))

    from collections import Counter

    print("Train label distribution:", Counter([dataset[i][1] for i in train_indices]))
    print("Val label distribution:", Counter([dataset[i][1] for i in val_indices]))
    print("Test label distribution:", Counter([dataset[i][1] for i in test_indices]))



    # Compute class weights
    train_labels = [dataset[idx][1] for idx in train_set.indices]
    label_counts = Counter(train_labels)
    total_samples = sum(label_counts.values())
    class_weights = [total_samples / label_counts[i] for i in range(len(label_counts))]

    # Device setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

    print(f"Class counts: {label_counts}")
    print(f"Class weights: {weights_tensor}")

    # Data loaders
    batch_size = 8
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    # Model setup
    model = HSHResNetPipeline().to(device)
    classifier_head = nn.Linear(2048, 2).to(device)
    optimizer = torch.optim.Adam(model.preprocessor.parameters(), lr= 1e-3)
    criterion = nn.CrossEntropyLoss(weight=weights_tensor)

    # Training
    train(model, classifier_head, train_loader, val_loader, device, optimizer, criterion, epochs = 500)

    # Evaluation
    test_loss, test_acc = evaluate(model, test_loader, classifier_head, device, criterion)
    print(f"Final Test Accuracy: {test_acc*100:.2f}%")
    # wandb.log({"final_test_accuracy": test_acc})

    # Save model
    torch.save(model.preprocessor.state_dict(), "hsh_preprocessor8.pth")
    print("Preprocessor weights saved to hsh_preprocessor5_batch8_split70_100epochs.pth")

    writer.close()



# if __name__ == "__main__":
#     root = "E:/PHD/Project/Papers/Journal_Oseberg/fragmentmatchingcode/hsh_data/"
#     dataset = FolderHSHDataset(root_dir=root)


#     wandb.init(project="fragment-matching", config={
#         "epochs": 50,
#         "batch_size": 8,
#         "learning_rate": 1e-3,
#         "architecture": "HSHResNetPipeline",
#         "dataset": "FolderHSHDataset"
#     })



#     total_len = len(dataset)
#     train_len = int(0.7 * total_len)
#     val_len = int(0.15 * total_len)
#     test_len = total_len - train_len - val_len

#     generator = torch.Generator().manual_seed(42)
#     train_set, val_set, test_set = random_split(dataset, [train_len, val_len, test_len], generator=generator)


#     train_labels = [dataset[idx][1] for idx in train_set.indices]  # get labels of training samples
#     label_counts = Counter(train_labels)
#     total_samples = sum(label_counts.values())

#     class_weights = [total_samples / label_counts[i] for i in range(len(label_counts))]
    
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     weights_tensor = torch.tensor(class_weights, dtype=torch.float).to(device)

#     print(f"Class counts: {label_counts}")
#     print(f"Class weights: {weights_tensor}")

#     train_loader = DataLoader(train_set, batch_size=8, shuffle=True)
#     val_loader = DataLoader(val_set, batch_size=8)
#     test_loader = DataLoader(test_set, batch_size=8)

    
#     model = HSHResNetPipeline().to(device)

#     classifier_head = nn.Linear(2048, 2).to(device)
#     optimizer = torch.optim.Adam(model.preprocessor.parameters(), lr=1e-3)
#     #criterion = nn.CrossEntropyLoss()
#     criterion = nn.CrossEntropyLoss(weight=weights_tensor)

#     # Train
#     #train(model, classifier_head, train_loader, val_loader, device, optimizer, criterion, 50)
#     train_losses, val_losses, train_accuracies, val_accuracies = train(
#     model, classifier_head, train_loader, val_loader, device, optimizer, criterion, 100 )

#     # Evaluate on test set
#     test_loss, test_acc = evaluate(model, test_loader, classifier_head, device, criterion)
#     print(f"Final Test Accuracy: {test_acc*100:.2f}%")

#         # Save trained preprocessor
#     torch.save(model.preprocessor.state_dict(), "hsh_preprocessor5_batch8_split70_100epochs.pth")
#     print("Preprocessor weights saved to hsh_preprocessor4.pth")

 