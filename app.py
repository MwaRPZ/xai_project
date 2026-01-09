import streamlit as st
import os
import torch
from torchvision import models, transforms
from torch import nn
from PIL import Image
import numpy as np
from modules.audio_detector.preprocess import create_spectrogram
from modules.audio_detector.lime_explainer import explain_audio_lime, visualize_lime
from modules.image_detector.grad_cam import GradCAM
from modules.common.shap_explainer import explain_with_shap, visualize_shap
import matplotlib.pyplot as plt

# Page Config
st.set_page_config(page_title="Unified XAI Platform", layout="wide")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# --- Helper: Load Models ---
@st.cache_resource
def load_audio_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    try:
        model.load_state_dict(torch.load('models/audio_deepfake_model.pth', map_location=device, weights_only=True))
        st.sidebar.success("✓ Audio model loaded")
    except Exception as e:
        st.sidebar.warning(f"⚠ Audio model not found: {e}")
    model.to(device)
    model.eval()
    return model

@st.cache_resource
def load_image_model():
    model = models.mobilenet_v2(weights=None)
    model.classifier[1] = nn.Linear(model.last_channel, 2)
    try:
        model.load_state_dict(torch.load('models/lung_cancer_model.pth', map_location=device, weights_only=True))
        st.sidebar.success("✓ Image model loaded")
    except Exception as e:
        st.sidebar.warning(f"⚠ Image model not found: {e}")
    model.to(device)
    model.eval()
    return model

audio_model = load_audio_model()
image_model = load_image_model()

# --- UI ---
st.title("🔬 Unified Explainable AI Platform")
st.markdown("**Deepfake Audio Detection** & **Lung Cancer Detection** with Advanced XAI")

# Sidebar
with st.sidebar:
    st.header("⚙️ Configuration")
    st.info("Models: MobileNetV2 (PyTorch)")
    
tab1, tab2, tab3 = st.tabs(["🎵 Deepfake Audio", "🫁 Lung Cancer", "📊 Comparison"])

# === Tab 1: Audio ===
with tab1:
    st.header("Deepfake Audio Detection")
    audio_file = st.file_uploader("Upload Audio File (.wav)", type=['wav'], key="aud")
    
    if audio_file:
        col1, col2 = st.columns([1, 2])
        
        with col1:
            st.audio(audio_file)
            
            # XAI Method Selection with Filtering
            st.subheader("XAI Methods")
            available_methods = ["LIME", "SHAP"]
            selected_xai = st.multiselect(
                "Select methods to apply:",
                available_methods,
                default=["LIME"],
                help="LIME and SHAP work on spectrogram representations"
            )
            
            if st.button("🔍 Analyze Audio", type="primary"):
                with st.spinner("Processing audio..."):
                    try:
                        # Save temp file
                        with open("temp.wav", "wb") as f:
                            f.write(audio_file.getbuffer())
                        
                        # Generate spectrogram
                        img_arr = create_spectrogram("temp.wav")
                        
                        # Predict
                        tens = torch.tensor(img_arr).permute(2, 0, 1).unsqueeze(0).float().to(device) / 255.0
                        
                        with torch.no_grad():
                            out = audio_model(tens)
                            probs = torch.softmax(out, dim=1)
                            conf, pred = torch.max(probs, 1)
                        
                        # Store in session state for comparison
                        st.session_state['audio_result'] = {
                            'prediction': pred.item(),
                            'confidence': conf.item(),
                            'tensor': tens,
                            'image_array': img_arr,
                            'selected_xai': selected_xai
                        }
                        
                        lbl = "FAKE" if pred.item() == 1 else "REAL"
                        st.success(f"✅ Analysis Complete!")
                        st.metric("Prediction", lbl, f"{conf.item():.1%}")
                        
                    except Exception as e:
                        st.error(f"Error processing audio: {e}")
        
        with col2:
            if 'audio_result' in st.session_state:
                result = st.session_state['audio_result']
                st.subheader("🔍 XAI Visualizations")
                
                # LIME
                if "LIME" in result['selected_xai']:
                    with st.expander("📌 LIME Explanation", expanded=True):
                        try:
                            expl = explain_audio_lime(result['image_array'].astype(np.double), audio_model, num_samples=100)
                            viz = visualize_lime(expl, result['prediction'])
                            st.image(viz, caption="LIME: Highlights important spectrogram regions", use_container_width=True)
                        except Exception as e:
                            st.error(f"LIME Error: {e}")
                
                # SHAP
                if "SHAP" in result['selected_xai']:
                    with st.expander("📊 SHAP Explanation", expanded=True):
                        try:
                            with st.spinner("Computing SHAP values..."):
                                shap_vals, base_val = explain_with_shap(audio_model, result['tensor'])
                                shap_viz = visualize_shap(shap_vals, result['tensor'], class_idx=result['prediction'])
                                st.image(shap_viz, caption="SHAP: Feature importance heatmap", use_container_width=True)
                        except Exception as e:
                            st.error(f"SHAP Error: {e}")

# === Tab 2: Image ===
with tab2:
    st.header("Lung Cancer Detection")
    img_file = st.file_uploader("Upload X-Ray Image", type=['png', 'jpg', 'jpeg'], key="img")
    
    if img_file:
        col3, col4 = st.columns([1, 2])
        
        with col3:
            image = Image.open(img_file).convert('RGB')
            st.image(image, caption="Uploaded X-Ray", width=250)
            
            # XAI Method Selection
            st.subheader("XAI Methods")
            img_xai_methods = ["Grad-CAM", "SHAP"]
            selected_img_xai = st.multiselect(
                "Select methods:",
                img_xai_methods,
                default=["Grad-CAM"],
                help="Grad-CAM shows spatial attention, SHAP shows feature importance"
            )
            
            if st.button("🔍 Analyze X-Ray", type="primary"):
                with st.spinner("Analyzing image..."):
                    try:
                        # Preprocess
                        t = transforms.Compose([
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
                        inp = t(image).unsqueeze(0).to(device)
                        
                        with torch.no_grad():
                            out = image_model(inp)
                            probs = torch.softmax(out, dim=1)
                            conf, pred = torch.max(probs, 1)
                        
                        # Store in session state
                        st.session_state['image_result'] = {
                            'prediction': pred.item(),
                            'confidence': conf.item(),
                            'tensor': inp,
                            'original_image': image,
                            'selected_xai': selected_img_xai
                        }
                        
                        lbl = "MALIGNANT" if pred.item() == 1 else "NORMAL"
                        st.success("✅ Analysis Complete!")
                        st.metric("Diagnosis", lbl, f"{conf.item():.1%}")
                        
                    except Exception as e:
                        st.error(f"Error analyzing image: {e}")
        
        with col4:
            if 'image_result' in st.session_state:
                result = st.session_state['image_result']
                st.subheader("🔍 XAI Visualizations")
                
                # Grad-CAM
                if "Grad-CAM" in result['selected_xai']:
                    with st.expander("🎯 Grad-CAM Heatmap", expanded=True):
                        try:
                            target_layer = image_model.features[-1]
                            cam = GradCAM(model=image_model, target_layer=target_layer)
                            heatmap = cam(result['tensor'], class_idx=result['prediction'])
                            
                            # Generate overlay
                            from PIL import Image as PILImage
                            hm_img = PILImage.fromarray(np.uint8(255 * heatmap))
                            hm_img = hm_img.resize((224, 224), resample=PILImage.BICUBIC)
                            hm_resized = np.array(hm_img) / 255.0
                            
                            cmap = plt.get_cmap("jet")
                            hm_colored = cmap(hm_resized)[:, :, :3]
                            hm_colored = np.uint8(255 * hm_colored)
                            
                            org = np.array(result['original_image'].resize((224, 224)))
                            overlay = 0.4 * hm_colored + 0.6 * org
                            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                            
                            st.image(overlay, caption="Grad-CAM: Regions influencing diagnosis", use_container_width=True)
                        except Exception as e:
                            st.error(f"Grad-CAM Error: {e}")
                
                # SHAP
                if "SHAP" in result['selected_xai']:
                    with st.expander("📊 SHAP Explanation", expanded=True):
                        try:
                            with st.spinner("Computing SHAP values..."):
                                shap_vals, base_val = explain_with_shap(image_model, result['tensor'])
                                shap_viz = visualize_shap(shap_vals, result['tensor'], class_idx=result['prediction'])
                                st.image(shap_viz, caption="SHAP: Feature importance", use_container_width=True)
                        except Exception as e:
                            st.error(f"SHAP Error: {e}")

# === Tab 3: Comparison ===
with tab3:
    st.header("📊 XAI Method Comparison")
    st.markdown("Compare different XAI methods side-by-side for the same input")
    
    # Check if we have results to compare
    has_audio = 'audio_result' in st.session_state
    has_image = 'image_result' in st.session_state
    
    if not has_audio and not has_image:
        st.info("👈 Analyze an audio file or image first to enable comparison")
    else:
        comparison_type = st.radio("Select data type:", ["Audio", "Image"], horizontal=True)
        
        if comparison_type == "Audio" and has_audio:
            result = st.session_state['audio_result']
            st.subheader(f"Prediction: {'FAKE' if result['prediction'] == 1 else 'REAL'} ({result['confidence']:.1%})")
            
            cols = st.columns(len(result['selected_xai']))
            
            for idx, method in enumerate(result['selected_xai']):
                with cols[idx]:
                    st.markdown(f"**{method}**")
                    try:
                        if method == "LIME":
                            expl = explain_audio_lime(result['image_array'].astype(np.double), audio_model, num_samples=100)
                            viz = visualize_lime(expl, result['prediction'])
                            st.image(viz, use_container_width=True)
                        elif method == "SHAP":
                            shap_vals, _ = explain_with_shap(audio_model, result['tensor'])
                            shap_viz = visualize_shap(shap_vals, result['tensor'], class_idx=result['prediction'])
                            st.image(shap_viz, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
        
        elif comparison_type == "Image" and has_image:
            result = st.session_state['image_result']
            st.subheader(f"Diagnosis: {'MALIGNANT' if result['prediction'] == 1 else 'NORMAL'} ({result['confidence']:.1%})")
            
            cols = st.columns(len(result['selected_xai']))
            
            for idx, method in enumerate(result['selected_xai']):
                with cols[idx]:
                    st.markdown(f"**{method}**")
                    try:
                        if method == "Grad-CAM":
                            target_layer = image_model.features[-1]
                            cam = GradCAM(model=image_model, target_layer=target_layer)
                            heatmap = cam(result['tensor'], class_idx=result['prediction'])
                            
                            from PIL import Image as PILImage
                            hm_img = PILImage.fromarray(np.uint8(255 * heatmap))
                            hm_img = hm_img.resize((224, 224), resample=PILImage.BICUBIC)
                            hm_resized = np.array(hm_img) / 255.0
                            
                            cmap = plt.get_cmap("jet")
                            hm_colored = cmap(hm_resized)[:, :, :3]
                            hm_colored = np.uint8(255 * hm_colored)
                            
                            org = np.array(result['original_image'].resize((224, 224)))
                            overlay = 0.4 * hm_colored + 0.6 * org
                            overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                            
                            st.image(overlay, use_container_width=True)
                        elif method == "SHAP":
                            shap_vals, _ = explain_with_shap(image_model, result['tensor'])
                            shap_viz = visualize_shap(shap_vals, result['tensor'], class_idx=result['prediction'])
                            st.image(shap_viz, use_container_width=True)
                    except Exception as e:
                        st.error(f"Error: {e}")
        else:
            st.warning(f"No {comparison_type.lower()} analysis available. Please analyze a file first.")

# Footer
st.markdown("---")
st.caption("🚀 Powered by PyTorch & Streamlit | Models trained on real datasets")
