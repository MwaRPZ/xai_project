import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import os
import pandas as pd
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # 2 classes: Normal (0), Abnormal (1)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    return model.to(device)

class CheXpertDataset(Dataset):
    def __init__(self, csv_file, root_dir, transform=None, max_samples=None):
        self.df = pd.read_csv(csv_file)
        if max_samples:
            self.df = self.df.sample(n=min(max_samples, len(self.df)), random_state=42)
        self.root_dir = root_dir
        self.transform = transform
        self.df.fillna(0, inplace=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        orig_path = self.df.iloc[idx]['Path']
        parts = orig_path.split('/')
        rel_path = os.path.join(*parts[1:])
        img_path = os.path.join(self.root_dir, rel_path)
        
        try:
            image = Image.open(img_path).convert("RGB")
        except:
            img_path = os.path.join(self.root_dir, orig_path)
            try:
                image = Image.open(img_path).convert("RGB")
            except:
                image = Image.new('RGB', (224, 224))

        if self.transform:
            image = self.transform(image)
        
        # Label Logic: "No Finding" == 1 -> Normal (0), else -> Abnormal (1)
        no_finding = self.df.iloc[idx]['No Finding']
        label = 0 if no_finding == 1.0 else 1
            
        return image, label

def train_lung_model(epochs=10, batch_size=32, max_samples=5000):
    csv_path = "data/image/train.csv"
    root_dir = "data/image"
    
    if not os.path.exists(csv_path):
        print(f"CSV not found at {csv_path}")
        return

    print("Loading CheXpert dataset...")
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    # Load full dataset
    full_dataset = CheXpertDataset(csv_path, root_dir, transform=transform, max_samples=max_samples)
    
    # Train/Val split (80/20)
    train_size = int(0.8 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(full_dataset, [train_size, val_size])
    
    print(f"Training samples: {train_size}, Validation samples: {val_size}")
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    print(f"\nStarting Image Training ({epochs} epochs)...")
    best_val_acc = 0.0
    
    for epoch in range(epochs):
        # Training
        model.train()
        train_loss = 0.0
        train_correct = 0
        train_total = 0
        
        for i, (images, targets) in enumerate(train_loader):
            images = images.to(device)
            targets = targets.long().to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            _, predicted = outputs.max(1)
            train_total += targets.size(0)
            train_correct += predicted.eq(targets).sum().item()
            
            if i % 10 == 0:
                print(f"Epoch {epoch+1}/{epochs}, Batch {i}/{len(train_loader)}, Loss: {loss.item():.4f}")
        
        train_acc = 100. * train_correct / train_total
        avg_train_loss = train_loss / len(train_loader)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_correct = 0
        val_total = 0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.long().to(device)
                
                outputs = model(images)
                loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()
        
        val_acc = 100. * val_correct / val_total
        avg_val_loss = val_loss / len(val_loader)
        
        print(f"\nEpoch {epoch+1}/{epochs}:")
        print(f"  Train Loss: {avg_train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {avg_val_loss:.4f}, Val Acc: {val_acc:.2f}%\n")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            os.makedirs('models', exist_ok=True)
            torch.save(model.state_dict(), 'models/lung_cancer_model.pth')
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    print(f"\nTraining Complete! Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_lung_model()
