import torch
import numpy as np
import shap
import matplotlib.pyplot as plt

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def explain_with_shap(model, image, num_samples=100):
    """
    Generate SHAP explanation for an image input.
    
    Args:
        model: PyTorch model
        image: Input image tensor (1, 3, 224, 224)
        num_samples: Number of samples for SHAP
        
    Returns:
        shap_values: SHAP values array
        base_value: Expected value
    """
    model.eval()
    
    # Create a wrapper function for SHAP
    def predict_fn(x):
        # x is numpy array (N, 3, 224, 224)
        with torch.no_grad():
            x_tensor = torch.tensor(x).float().to(device)
            outputs = model(x_tensor)
            probs = torch.softmax(outputs, dim=1)
        return probs.cpu().numpy()
    
    # Convert image to numpy for SHAP
    img_np = image.cpu().numpy()
    
    # Create background dataset (use mean image or zeros)
    background = np.zeros((1, 3, 224, 224))
    
    # Create SHAP explainer
    explainer = shap.KernelExplainer(predict_fn, background)
    
    # Calculate SHAP values
    shap_values = explainer.shap_values(img_np, nsamples=num_samples)
    
    return shap_values, explainer.expected_value

def visualize_shap(shap_values, image, class_idx=1):
    """
    Visualize SHAP values as a heatmap overlay.
    
    Args:
        shap_values: SHAP values from explain_with_shap
        image: Original image tensor (1, 3, 224, 224)
        class_idx: Class index to visualize
        
    Returns:
        PIL Image with SHAP overlay
    """
    # Get SHAP values for the predicted class
    if isinstance(shap_values, list):
        shap_vals = shap_values[class_idx]
    else:
        shap_vals = shap_values
    
    # Sum across channels to get spatial importance
    # shap_vals shape: (1, 3, 224, 224)
    shap_spatial = np.abs(shap_vals[0]).sum(axis=0)  # (224, 224)
    
    # Normalize
    shap_spatial = (shap_spatial - shap_spatial.min()) / (shap_spatial.max() - shap_spatial.min() + 1e-8)
    
    # Create heatmap
    from PIL import Image as PILImage
    
    # Apply colormap
    cmap = plt.get_cmap("hot")
    heatmap_colored = cmap(shap_spatial)[:, :, :3]
    heatmap_colored = np.uint8(255 * heatmap_colored)
    
    # Original image (denormalize if needed)
    img_np = image[0].cpu().permute(1, 2, 0).numpy()
    
    # If normalized, denormalize
    # Assuming ImageNet normalization
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    img_np = np.uint8(255 * img_np)
    
    # Overlay
    overlay = 0.5 * heatmap_colored + 0.5 * img_np
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return PILImage.fromarray(overlay)
