import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
import os
import io
from PIL import Image

def create_spectrogram(audio_path, target_size=(224, 224)):
    """
    Generates a Mel-Spectrogram image from an audio file.
    Returns the image array (224, 224, 3) suitable for MobileNet.
    """
    y, sr = librosa.load(audio_path)
    ms = librosa.feature.melspectrogram(y=y, sr=sr)
    log_ms = librosa.power_to_db(ms, ref=np.max)

    # Plot to buffer to avoid saving to disk if possible
    # We use a non-interactive backend to facilitate running in threads/servers
    plt.switch_backend('Agg') 
    
    plt.figure(figsize=(10, 10))
    fig = plt.gcf()
    ax = plt.gca()
    ax.axis('off') 
    
    librosa.display.specshow(log_ms, sr=sr)
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    
    # Save to memory buffer
    buf = io.BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    
    # Load from buffer using PIL
    img = Image.open(buf).convert('RGB')
    img = img.resize(target_size)
    
    # Convert to numpy array
    img_array = np.array(img)
    
    # In Keras img_to_array makes it float 0-255 maybe?
    # Here we stick to uint8 0-255 or convert to float in training loop
    # We return numpy array of shape (224, 224, 3)
    
    return img_array

def preprocess_audio_for_model(audio_path):
    # This was a Keras Helper. 
    # For PyTorch we usually do this in the Dataset class or transform.
    img = create_spectrogram(audio_path)
    # Return as is, let DataLoader handle tensor conversion
    return img
