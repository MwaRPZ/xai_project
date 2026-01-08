import kagglehub
import os
import shutil

# Create data directory if it doesn't exist
os.makedirs("data/audio", exist_ok=True)
os.makedirs("data/image", exist_ok=True)

print("Downloading Deepfake Audio Dataset...")
audio_path = kagglehub.dataset_download("mohammedabdeldayem/the-fake-or-real-dataset")
print(f"Audio dataset downloaded to: {audio_path}")

# Symlink or Copy (Symbolic link is better to save space, but copy is safer for permissions on Windows sometimes. Let's try to just print path first)
# We will just print the path for now and I will manually move/link them later or use the path directly.

print("Downloading Lung Cancer Dataset...")
image_path = kagglehub.dataset_download("ashery/chexpert")
print(f"Lung Cancer dataset downloaded to: {image_path}")
