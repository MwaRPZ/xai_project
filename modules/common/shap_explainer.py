import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from PIL import Image as PILImage
import gc

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def explain_with_shap(model, image):
    model.eval()
    
    try:
        image_with_grad = image.clone().detach().requires_grad_(True)
        background = torch.zeros(1, 3, 224, 224).to(device).requires_grad_(True)
        
        explainer = shap.GradientExplainer(model, background)
        shap_values = explainer.shap_values(image_with_grad)
        
        if torch.is_tensor(shap_values):
            shap_values = shap_values.detach().cpu().numpy()
        
        if isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 5:
            shap_class0 = shap_values[:, :, :, :, 0]
            shap_class1 = shap_values[:, :, :, :, 1]
            shap_values = [shap_class0, shap_class1]
        elif isinstance(shap_values, np.ndarray) and len(shap_values.shape) == 4:
            if shap_values.shape[0] == 2:
                shap_values = [shap_values[0:1], shap_values[1:2]]
            else:
                shap_values = [shap_values, shap_values]
        
        if not isinstance(shap_values, list):
            shap_values = [shap_values, shap_values]
        
        del explainer, background, image_with_grad
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        
        return shap_values, None
        
    except Exception as e:
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        gc.collect()
        raise e

def visualize_shap(shap_values, image, class_idx=1):
    try:
        if isinstance(shap_values, list):
            class_idx = min(class_idx, len(shap_values) - 1)
            shap_vals = shap_values[class_idx]
        else:
            shap_vals = shap_values
        
        if torch.is_tensor(shap_vals):
            shap_vals = shap_vals.cpu().numpy()
        
        if shap_vals.ndim == 4 and shap_vals.shape[0] == 1:
            shap_vals = shap_vals[0]
        
        if shap_vals.shape != (3, 224, 224):
            if shap_vals.size == 3 * 224 * 224:
                shap_vals = shap_vals.reshape(3, 224, 224)
            else:
                raise ValueError(f"Unexpected shape: {shap_vals.shape}")
        
        shap_spatial = np.abs(shap_vals).sum(axis=0)
        
        shap_min = shap_spatial.min()
        shap_max = shap_spatial.max()
        
        if shap_max > shap_min:
            shap_spatial = (shap_spatial - shap_min) / (shap_max - shap_min)
        else:
            shap_spatial = np.zeros_like(shap_spatial)
        
        cmap = plt.get_cmap("hot")
        heatmap_colored = cmap(shap_spatial)[:, :, :3]
        heatmap_colored = np.uint8(255 * heatmap_colored)
        
        img_np = image[0].cpu().permute(1, 2, 0).numpy()
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])
        img_np = std * img_np + mean
        img_np = np.clip(img_np, 0, 1)
        img_np = np.uint8(255 * img_np)
        
        overlay = 0.5 * heatmap_colored + 0.5 * img_np
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        
        del shap_vals, shap_spatial, heatmap_colored, img_np
        gc.collect()
        
        return PILImage.fromarray(overlay)
        
    except Exception as e:
        gc.collect()
        raise e