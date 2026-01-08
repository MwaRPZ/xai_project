# Quick Setup Guide for Teammates

## Prerequisites
- Python 3.14+ installed
- Git installed
- (Optional) Kaggle account for downloading datasets

## Setup Steps

### 1. Clone the Repository
```bash
git clone <repository-url>
cd XAI
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Data (Optional - for training)
```bash
# You'll need a Kaggle account
python download_data.py
```

### 4. Run the Application
```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`

## Project Structure
```
XAI/
├── app.py                  # Main Streamlit app - START HERE
├── modules/
│   ├── audio_detector/     # Audio deepfake detection
│   ├── image_detector/     # Lung cancer detection
│   └── common/             # Shared XAI utilities
├── models/                 # Trained models (not in git)
└── data/                   # Datasets (not in git)
```

## Training Models (Optional)
If you want to train your own models:

```bash
# Audio model (5 epochs, ~30-60 min)
python modules/audio_detector/train.py

# Image model (10 epochs, ~1-2 hours)
python modules/image_detector/train.py
```

## Features
- 🎵 Deepfake Audio Detection (LIME + SHAP)
- 🫁 Lung Cancer Detection (Grad-CAM + SHAP)
- 📊 Side-by-side XAI comparison

## Troubleshooting

**Issue**: `ModuleNotFoundError`
- **Fix**: Run `pip install -r requirements.txt`

**Issue**: Models not found
- **Fix**: Either train models or use the app in demo mode (it will show the UI)

**Issue**: SHAP is slow
- **Fix**: Reduce `num_samples` parameter in the code (default is 50)

## Contributing
1. Create a new branch: `git checkout -b feature/your-feature`
2. Make changes and commit: `git commit -am "Add feature"`
3. Push: `git push origin feature/your-feature`
4. Create a Pull Request

## Questions?
Check the main [README.md](README.md) for detailed documentation.
