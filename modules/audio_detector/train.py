import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import os
import glob
import numpy as np
from preprocess import create_spectrogram
from PIL import Image

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def build_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # 2 classes: Real (0), Fake (1)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    return model.to(device)

class AudioDataset(Dataset):
    def __init__(self, file_paths, labels, transform=None):
        self.file_paths = file_paths
        self.labels = labels

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        label = self.labels[idx]
        
        try:
            # Generate spectrogram on the fly
            img_arr = create_spectrogram(path)
            
            # Convert to Tensor (C, H, W)
            img_tensor = torch.tensor(img_arr).permute(2, 0, 1).float() / 255.0
            
            return img_tensor, label
        except Exception as e:
            print(f"Error loading {path}: {e}")
            return torch.zeros(3, 224, 224), label

def train_audio_model(data_dir="data/audio", epochs=5, batch_size=16):
    print(f"Searching for audio files in {data_dir}...")
    
    # Recursive search
    all_files = glob.glob(os.path.join(data_dir, '**', '*.wav'), recursive=True)
    
    real_files = [f for f in all_files if "original" in f.lower()]
    fake_files = [f for f in all_files if "rerecorded" in f.lower()]
    
    print(f"Found {len(real_files)} Real and {len(fake_files)} Fake files.")
    
    if not real_files and not fake_files:
        print("No audio files found! Check directory structure.")
        return

    # Limit for reasonable training time (optional)
    max_per_class = 2000
    if len(real_files) > max_per_class:
        real_files = np.random.choice(real_files, max_per_class, replace=False).tolist()
    if len(fake_files) > max_per_class:
        fake_files = np.random.choice(fake_files, max_per_class, replace=False).tolist()
    
    files = real_files + fake_files
    labels = [0]*len(real_files) + [1]*len(fake_files)
    
    # Shuffle
    combined = list(zip(files, labels))
    np.random.shuffle(combined)
    files[:], labels[:] = zip(*combined)
    
    # Train/Val split (80/20)
    split_idx = int(0.8 * len(files))
    train_files, val_files = files[:split_idx], files[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]
    
    print(f"Training samples: {len(train_files)}, Validation samples: {len(val_files)}")
    
    train_dataset = AudioDataset(train_files, train_labels)
    val_dataset = AudioDataset(val_files, val_labels)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    
    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
    
    print(f"\nStarting Audio Training ({epochs} epochs)...")
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
            
            if i % 20 == 0:
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
            torch.save(model.state_dict(), 'models/audio_deepfake_model.pth')
            print(f"✓ Saved best model (Val Acc: {val_acc:.2f}%)")
    
    print(f"\nTraining Complete! Best Validation Accuracy: {best_val_acc:.2f}%")

if __name__ == "__main__":
    train_audio_model()
