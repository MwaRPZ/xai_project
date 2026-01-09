import torch
import numpy as np
import shap
import matplotlib.pyplot as plt
from PIL import Image as PILImage

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def explain_with_shap(model, image, num_samples=50):
    """
    Generate SHAP explanation using GradientExplainer.
    
    Args:
        model: PyTorch model
        image: Input image tensor (1, 3, 224, 224) - NORMALIZED
        num_samples: Number of samples
        
    Returns:
        shap_values: SHAP values array (format normalisé)
        base_value: None
    """
    model.eval()
    
    # Background simple
    background = torch.zeros(1, 3, 224, 224).to(device)
    
    # Créer GradientExplainer
    explainer = shap.GradientExplainer(model, background)
    
    # Calculer SHAP values
    shap_values = explainer.shap_values(image)
    
    # Debug
    print(f"[SHAP] Type: {type(shap_values)}")
    
    # Convertir en numpy si tensor
    if torch.is_tensor(shap_values):
        shap_values = shap_values.cpu().numpy()
    
    print(f"[SHAP] Shape brute: {shap_values.shape}")
    
    # ===== Normaliser le format =====
    # Format attendu par GradientExplainer pour 2 classes :
    # (1, 3, 224, 224, 2) où le dernier 2 = [classe0, classe1]
    
    if isinstance(shap_values, np.ndarray):
        if len(shap_values.shape) == 5:
            # Shape: (1, 3, 224, 224, 2)
            # Séparer les classes
            shap_class0 = shap_values[:, :, :, :, 0]  # (1, 3, 224, 224)
            shap_class1 = shap_values[:, :, :, :, 1]  # (1, 3, 224, 224)
            shap_values = [shap_class0, shap_class1]
            print(f"[SHAP] Format 5D détecté, séparation en 2 classes")
            print(f"[SHAP] Classe 0: {shap_class0.shape}, Classe 1: {shap_class1.shape}")
        elif len(shap_values.shape) == 4:
            # Shape: (2, 3, 224, 224) ou (1, 3, 224, 224)
            if shap_values.shape[0] == 2:
                shap_values = [shap_values[0:1], shap_values[1:2]]
                print(f"[SHAP] Format 4D avec 2 classes détecté")
            else:
                # Une seule classe, dupliquer
                shap_values = [shap_values, shap_values]
                print(f"[SHAP] Format 4D avec 1 classe, duplication")
    
    # Si c'est déjà une liste
    if not isinstance(shap_values, list):
        shap_values = [shap_values, shap_values]
        print(f"[SHAP] Conversion en liste par défaut")
    
    print(f"[SHAP] Format final: liste de {len(shap_values)} classes")
    
    return shap_values, None

def visualize_shap(shap_values, image, class_idx=1):
    """
    Visualize SHAP values as a heatmap overlay.
    
    Args:
        shap_values: SHAP values (liste [class0, class1])
        image: Original image tensor (1, 3, 224, 224) - NORMALIZED
        class_idx: Class index (0 ou 1)
        
    Returns:
        PIL Image with SHAP overlay
    """
    # Extraire la classe
    if isinstance(shap_values, list):
        class_idx = min(class_idx, len(shap_values) - 1)
        shap_vals = shap_values[class_idx]
    else:
        shap_vals = shap_values
    
    # Convertir en numpy
    if torch.is_tensor(shap_vals):
        shap_vals = shap_vals.cpu().numpy()
    
    print(f"[VIZ] Shape après extraction classe {class_idx}: {shap_vals.shape}")
    
    # On veut (3, 224, 224)
    if shap_vals.ndim == 4 and shap_vals.shape[0] == 1:
        # (1, 3, 224, 224) → (3, 224, 224)
        shap_vals = shap_vals[0]
        print(f"[VIZ] Suppression dimension batch: {shap_vals.shape}")
    
    # Vérification finale
    if shap_vals.shape != (3, 224, 224):
        print(f"[VIZ] ⚠️ Shape inattendue: {shap_vals.shape}, tentative de correction...")
        # Dernier recours
        if shap_vals.size == 3 * 224 * 224:
            shap_vals = shap_vals.reshape(3, 224, 224)
            print(f"[VIZ] Reshape forcé réussi: {shap_vals.shape}")
        else:
            raise ValueError(f"Impossible de reshaper {shap_vals.shape} en (3, 224, 224)")
    
    # Agréger sur les channels RGB
    shap_spatial = np.abs(shap_vals).sum(axis=0)  # (224, 224)
    print(f"[VIZ] Shape après agrégation RGB: {shap_spatial.shape}")
    
    # Normaliser
    shap_min = shap_spatial.min()
    shap_max = shap_spatial.max()
    
    if shap_max > shap_min:
        shap_spatial = (shap_spatial - shap_min) / (shap_max - shap_min)
    else:
        shap_spatial = np.zeros_like(shap_spatial)
    
    # Colormap
    cmap = plt.get_cmap("hot")
    heatmap_colored = cmap(shap_spatial)[:, :, :3]
    heatmap_colored = np.uint8(255 * heatmap_colored)
    
    # Dénormaliser l'image originale
    img_np = image[0].cpu().permute(1, 2, 0).numpy()
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img_np = std * img_np + mean
    img_np = np.clip(img_np, 0, 1)
    img_np = np.uint8(255 * img_np)
    
    # Overlay
    overlay = 0.5 * heatmap_colored + 0.5 * img_np
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)
    
    print(f"[VIZ] ✅ Overlay créé avec succès: {overlay.shape}")
    
    return PILImage.fromarray(overlay)
