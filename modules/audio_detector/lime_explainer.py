from lime import lime_image
import numpy as np
import torch
from skimage.segmentation import mark_boundaries

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_predict_fn(model):
    def predict(images):
        # Images comes as (N, 224, 224, 3) double numpy
        # PyTorch expects (N, 3, 224, 224) float tensor
        model.eval()
        
        # Transpose to Channel First
        images = np.transpose(images, (0, 3, 1, 2)).astype(np.float32)
        images = torch.tensor(images).to(device)
        
        with torch.no_grad():
            outputs = model(images)
            probs = torch.nn.functional.softmax(outputs, dim=1)
            
        return probs.cpu().numpy()
    return predict

def explain_audio_lime(img_array, model, num_samples=1000):
    explainer = lime_image.LimeImageExplainer()
    predict_fn = get_predict_fn(model)
    
    # Simple check for double
    if img_array.dtype != np.double:
        img_array = img_array.astype(np.double)

    explanation = explainer.explain_instance(
        img_array, 
        predict_fn, 
        hide_color=0, 
        num_samples=num_samples
    )
    return explanation

def visualize_lime(explanation, label):
    # Récupérer l'image et le masque
    temp, mask = explanation.get_image_and_mask(
        label,
        positive_only=False,
        num_features=5,
        hide_rest=False,
    )

    # Normaliser temp dans [0, 1] au lieu de temp/2+0.5
    temp = temp.astype(np.float32)
    temp = (temp - temp.min()) / (temp.max() - temp.min() + 1e-8)

    img_boundary = mark_boundaries(temp, mask)
    return img_boundary
