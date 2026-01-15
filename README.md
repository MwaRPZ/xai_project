# Unified XAI Interface for Deepfake & Lung Cancer Detection

## Authors

- **Adrien Servas**
- **Alexandre Francony**
- **Léonard Seidlitz**
- **Raphael Roux**
- **Romain Requena**

## Use of AI
We used AI in this project to :
- Help us with the code implementation of XAI models
- Help us with the project structure
- Help us with debugging

The AI that we used are Chat GPT and Gemini

This project integrates two Explainable AI (XAI) systems into a single Streamlit interface:
1.  **Deepfake Audio Detection**: MobileNetV2 + LIME + SHAP
2.  **Lung Cancer Detection**: MobileNetV2 + Grad-CAM + SHAP

## ✨ Features

### Core Functionality
- **Dual-Mode Analysis**: Audio deepfake detection and medical image diagnosis
- **Multiple XAI Methods**: LIME, SHAP, and Grad-CAM with automatic filtering
- **Side-by-Side Comparison**: Compare different XAI methods on the same input
- **Real-Time Inference**: PyTorch-based models with GPU acceleration support

### XAI Techniques
- **LIME (Local Interpretable Model-agnostic Explanations)**: Highlights important regions
- **SHAP (SHapley Additive exPlanations)**: Feature importance visualization
- **Grad-CAM (Gradient-weighted Class Activation Mapping)**: Spatial attention heatmaps

## 📁 Repository Structure
```
XAI_Project/
├── app.py                      # Main Streamlit Application
├── requirements.txt            # Python dependencies
├── download_data.py            # Dataset download script
├── modules/
│   ├── audio_detector/         # Audio preprocessing & training
│   │   ├── train.py           # Enhanced training (5 epochs, validation)
│   │   ├── preprocess.py      # Spectrogram generation
│   │   └── lime_explainer.py  # LIME for audio
│   ├── image_detector/         # Image preprocessing & training
│   │   ├── train.py           # Enhanced training (10 epochs, validation)
│   │   └── grad_cam.py        # Grad-CAM implementation
│   └── common/
│       └── shap_explainer.py  # SHAP for both modalities
├── models/                     # Trained model weights (.pth)
└── data/                       # Datasets (linked to Kaggle cache)
```

## 🚀 Setup Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

**Note**: This project requires **Python 3.14+** and uses **PyTorch** (not TensorFlow).

### 2. Download Datasets (Optional)
```bash
python download_data.py
```
*Downloads ~4GB+ of data from Kaggle. Required only for training.*

### 3. Train Models (Optional)
```bash
# Train Audio Model (5 epochs, ~30-60 min)
python modules/audio_detector/train.py

# Train Image Model (10 epochs, ~1-2 hours)
python modules/image_detector/train.py
```

### 4. Run the Application
```bash
streamlit run app.py
```

## 🎯 Usage

1. **Select a Tab**: Choose "Deepfake Audio" or "Lung Cancer"
2. **Upload File**: Upload a `.wav` audio file or `.jpg/.png` X-ray image
3. **Select XAI Methods**: Choose which explanation techniques to apply
4. **Analyze**: Click the analyze button to get predictions and explanations
5. **Compare**: Use the "Comparison" tab to view XAI methods side-by-side

## 🔧 Technical Details

### Models
- **Architecture**: MobileNetV2 (transfer learning from ImageNet)
- **Audio Classes**: Real (0) vs Fake (1)
- **Image Classes**: Normal (0) vs Malignant (1)

### Training
- **Audio**: 5 epochs, 80/20 train/val split, Adam optimizer
- **Image**: 10 epochs, 80/20 train/val split, Adam optimizer
- **Validation**: Best model saved based on validation accuracy

### Data Processing
- **Audio**: Converted to Mel-spectrograms (224×224×3)
- **Image**: Resized to 224×224, ImageNet normalization

## 📊 Datasets
- **Audio**: [Fake or Real Dataset](https://www.kaggle.com/datasets/mohammedabdeldayem/the-fake-or-real-dataset)
- **Image**: [CheXpert Dataset](https://www.kaggle.com/datasets/ashery/chexpert)

## 🎨 Design Decisions
- **PyTorch Backend**: Chosen for Python 3.14+ compatibility (TensorFlow not supported)
- **No OpenCV**: Replaced with PIL/Matplotlib for broader compatibility
- **Automatic XAI Filtering**: Methods are pre-filtered based on input modality
- **Session State**: Results cached for comparison functionality

## 📝 Project Status
✅ All critical and important features implemented
✅ Full PyTorch migration complete
✅ Enhanced training with validation
✅ SHAP integration for both modalities
✅ Functional comparison tab
✅ XAI method filtering

## 🤝 Contributing
This is an academic project for XAI coursework.

## 📄 License
Educational use only.

