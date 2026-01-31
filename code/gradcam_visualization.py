"""
GradCAM Visualization for HSANet
================================
Generates interpretability visualizations showing where the model focuses.

Author: HSANet Team
Date: January 2026
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import cv2
from pathlib import Path
import torchvision.transforms as T


class GradCAM:
    """GradCAM for model interpretability"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        # Register hooks
        self.forward_hook = target_layer.register_forward_hook(self._save_activation)
        self.backward_hook = target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()
    
    def generate(self, input_tensor, target_class=None):
        """Generate GradCAM heatmap"""
        self.model.eval()
        
        # Forward pass
        outputs = self.model(input_tensor)
        probs = outputs['probs']
        
        if target_class is None:
            target_class = probs.argmax(dim=1).item()
        
        # Backward pass
        self.model.zero_grad()
        one_hot = torch.zeros_like(probs)
        one_hot[0, target_class] = 1
        probs.backward(gradient=one_hot, retain_graph=True)
        
        # Generate CAM
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().numpy()
        
        # Global average pooling of gradients
        weights = np.mean(gradients, axis=(1, 2))
        
        # Weighted combination of activation maps
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        # ReLU and normalize
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (224, 224))
        if cam.max() > 0:
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        
        return cam, target_class
    
    def remove_hooks(self):
        self.forward_hook.remove()
        self.backward_hook.remove()


def visualize_predictions(model, image_paths, class_names, device='cuda', save_dir=None):
    """Visualize GradCAM for multiple images"""
    if save_dir:
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
    
    # Get target layer (last conv block in backbone)
    target_layer = model.backbone.blocks[-1]
    gradcam = GradCAM(model, target_layer)
    
    # Transform for model input
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    fig, axes = plt.subplots(len(image_paths), 4, figsize=(16, 4 * len(image_paths)))
    if len(image_paths) == 1:
        axes = axes.reshape(1, -1)
    
    for idx, img_path in enumerate(image_paths):
        # Load image
        original = Image.open(img_path).convert('RGB')
        input_tensor = transform(original).unsqueeze(0).to(device)
        
        # Get prediction and GradCAM
        with torch.enable_grad():
            cam, pred_class = gradcam.generate(input_tensor)
        
        # Get confidence and uncertainty
        with torch.no_grad():
            outputs = model(input_tensor)
            confidence = outputs['probs'][0, pred_class].item()
            uncertainty = outputs['uncertainty_total'][0].item()
        
        # Resize original for overlay
        original_resized = np.array(original.resize((224, 224))) / 255.0
        
        # Create heatmap
        heatmap = plt.cm.jet(cam)[:, :, :3]
        
        # Create overlay
        overlay = 0.6 * original_resized + 0.4 * heatmap
        
        # Plot
        axes[idx, 0].imshow(original_resized)
        axes[idx, 0].set_title(f'Original\nTrue: {Path(img_path).parent.name}')
        axes[idx, 0].axis('off')
        
        axes[idx, 1].imshow(cam, cmap='jet')
        axes[idx, 1].set_title('GradCAM Heatmap')
        axes[idx, 1].axis('off')
        
        axes[idx, 2].imshow(overlay)
        axes[idx, 2].set_title(f'Overlay\nPred: {class_names[pred_class]}')
        axes[idx, 2].axis('off')
        
        # Prediction info
        axes[idx, 3].text(0.5, 0.6, f'Predicted: {class_names[pred_class]}', 
                         ha='center', va='center', fontsize=12, fontweight='bold')
        axes[idx, 3].text(0.5, 0.4, f'Confidence: {confidence:.2%}', 
                         ha='center', va='center', fontsize=11)
        axes[idx, 3].text(0.5, 0.25, f'Uncertainty: {uncertainty:.4f}', 
                         ha='center', va='center', fontsize=11)
        axes[idx, 3].set_xlim(0, 1)
        axes[idx, 3].set_ylim(0, 1)
        axes[idx, 3].axis('off')
    
    plt.tight_layout()
    
    if save_dir:
        plt.savefig(save_dir / 'gradcam_visualization.png', dpi=150, bbox_inches='tight')
    
    plt.show()
    
    gradcam.remove_hooks()


def visualize_uncertainty_distribution(model, dataloader, class_names, device='cuda', save_path=None):
    """Visualize uncertainty distribution for correct vs incorrect predictions"""
    model.eval()
    
    correct_unc = []
    incorrect_unc = []
    class_uncertainties = {i: [] for i in range(len(class_names))}
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            
            preds = outputs['probs'].argmax(dim=1)
            uncertainties = outputs['uncertainty_total']
            
            for i in range(len(labels)):
                unc = uncertainties[i].item()
                true_class = labels[i].item()
                pred_class = preds[i].item()
                
                class_uncertainties[true_class].append(unc)
                
                if pred_class == true_class:
                    correct_unc.append(unc)
                else:
                    incorrect_unc.append(unc)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Correct vs Incorrect
    axes[0].hist(correct_unc, bins=30, alpha=0.7, label=f'Correct (n={len(correct_unc)})', color='green')
    if incorrect_unc:
        axes[0].hist(incorrect_unc, bins=30, alpha=0.7, label=f'Incorrect (n={len(incorrect_unc)})', color='red')
    axes[0].set_xlabel('Uncertainty')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Uncertainty: Correct vs Incorrect Predictions')
    axes[0].legend()
    
    # Per-class uncertainty
    positions = list(range(len(class_names)))
    bp = axes[1].boxplot([class_uncertainties[i] for i in range(len(class_names))], 
                         positions=positions, patch_artist=True)
    axes[1].set_xticks(positions)
    axes[1].set_xticklabels(class_names, rotation=45)
    axes[1].set_ylabel('Uncertainty')
    axes[1].set_title('Uncertainty Distribution by Class')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    
    plt.show()
    
    # Print statistics
    print("\nUncertainty Statistics:")
    print(f"  Correct predictions: {np.mean(correct_unc):.4f} ± {np.std(correct_unc):.4f}")
    if incorrect_unc:
        print(f"  Incorrect predictions: {np.mean(incorrect_unc):.4f} ± {np.std(incorrect_unc):.4f}")


if __name__ == "__main__":
    print("GradCAM visualization module loaded.")
    print("Usage:")
    print("  from gradcam_visualization import visualize_predictions, GradCAM")
    print("  visualize_predictions(model, image_paths, class_names, device)")
