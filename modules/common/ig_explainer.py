import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image as PILImage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def explain_with_ig(model, input_tensor, baseline=None, steps=50):
    """
    Compute Integrated Gradients for a given input.
    
    Args:
        model: PyTorch model (must return logits)
        input_tensor: Input image tensor (1, 3, 224, 224)
        baseline: Baseline tensor (default: zeros)
        steps: Number of interpolation steps
        
    Returns:
        attributions: Tensor of same shape as input with attribution scores
        prediction_score: Confidence score of the target class
        target_class: The predicted class index
    """
    model.eval()
    
    # 1. Get Model Prediction to determine target class
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output, dim=1)
        prediction_score, target_class = torch.max(probabilities, 1)
        target_class_idx = target_class.item()
    
    # 2. Define Baseline (Black image if None)
    if baseline is None:
        baseline = torch.zeros_like(input_tensor).to(device)
        
    # 3. Generate Interpolated Inputs
    # Shape: (steps + 1, 3, 224, 224)
    alphas = torch.linspace(0, 1, steps + 1).to(device)
    alphas = alphas.view(-1, 1, 1, 1) # Broadcastable shape
    
    # Interpolation: baseline + alpha * (input - baseline)
    interpolated_inputs = baseline + alphas * (input_tensor - baseline)
    interpolated_inputs.requires_grad_(True)
    
    # 4. Compute Gradients for all interpolated steps
    # We may need to process in batches if memory is tight, but 50 steps usually fits
    attributions_accum = 0
    
    # Passing the whole batch of 51 images through the model at once might be heavy
    # Let's simple loop or small batch if needed. For now, try full batch.
    
    # Forward pass
    outputs = model(interpolated_inputs)
    
    # We want gradients w.r.t the TARGET class
    target_score = outputs[:, target_class_idx].sum()
    
    # Backward pass
    model.zero_grad()
    target_score.backward()
    
    gradients = interpolated_inputs.grad
    
    # 5. Integral Approximation (Riemann Sum)
    # Average gradients across all steps (approximate integral)
    avg_gradients = torch.mean(gradients[:-1], dim=0, keepdim=True) # Exclude last point usually, or simple mean
    avg_gradients = torch.mean(gradients, dim=0, keepdim=True) # Using simple mean of all points
    
    # 6. Integrated Gradients = (Input - Baseline) * Avg_Gradients
    integrated_gradients = (input_tensor - baseline) * avg_gradients
    
    return integrated_gradients, prediction_score.item(), target_class_idx

def visualize_ig(attributions, original_image):
    """
    Visualize Integrated Gradients as a heatmap.
    
    Args:
        attributions: Tensor (1, 3, 224, 224)
        original_image: Tensor (1, 3, 224, 224)
        
    Returns:
        PIL Image overlay
    """
    # 1. Convert to Numpy
    attr = attributions[0].cpu().detach().numpy() # (3, 224, 224)
    
    # 2. Aggregate across channels
    # We typically sum or take max absolute value across RGB
    attr = np.sum(np.abs(attr), axis=0) # (224, 224)
    
    # 3. Normalize for visualization
    # Robust normalization (like percentile) handles outliers better than min/max
    vmax = np.percentile(np.abs(attr), 99)
    vmin = np.min(attr)
    
    # Clip and scale to 0-1
    attr = np.clip(attr, 0, vmax)
    attr = (attr - vmin) / (vmax - vmin + 1e-8)
    
    # 4. Color Map (Inferno or Fire is often good for "Intensity")
    cmap = plt.get_cmap("inferno")
    heatmap = cmap(attr)[:, :, :3] # (224, 224, 3)
    heatmap = np.uint8(255 * heatmap)
    
    # 5. Prepare Original Image
    img_np = original_image[0].cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    img_np = np.uint8(255 * img_np)
    
    # 6. Overlay
    overlay = 0.6 * heatmap + 0.4 * img_np
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    return PILImage.fromarray(overlay)
