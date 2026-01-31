"""
Cross-Dataset Validation for HSANet
====================================
This script evaluates HSANet on external datasets to demonstrate generalizability.

External Datasets Supported:
1. Figshare Brain Tumor Dataset (3,064 images)
2. BraTS Challenge Dataset (needs preprocessing)
3. Custom external datasets

This validation adds significant novelty by proving:
- Model generalization to unseen data distributions
- Robustness to different scanner/acquisition protocols
- Clinical readiness for real-world deployment

Author: HSANet Team
Date: January 2026
"""

import os
import sys
import json
import random
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import argparse

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from PIL import Image
import torchvision.transforms as T

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, cohen_kappa_score, matthews_corrcoef,
    roc_auc_score, roc_curve, auc, classification_report
)
from scipy import stats

warnings.filterwarnings('ignore')

# Import HSANet model
from hsanet_model import HSANetV2, create_model


# ============================================================================
# Configuration
# ============================================================================

class CrossValidationConfig:
    """Configuration for cross-dataset validation"""
    
    # Class mapping (ensure consistency across datasets)
    CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    NUM_CLASSES = 4
    
    # Image settings (must match training)
    IMG_SIZE = 224
    NORMALIZE_MEAN = [0.485, 0.456, 0.406]
    NORMALIZE_STD = [0.229, 0.224, 0.225]
    
    # Paths
    PRETRAINED_MODEL = Path("./hsanet_results/hsanet_final.pth")
    OUTPUT_DIR = Path("./cross_validation_results")
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    BATCH_SIZE = 32
    
    # Dataset paths (update these for your setup)
    DATASETS = {
        'Kaggle_Original': Path("./brain-tumor-mri-dataset/Testing"),
        'Figshare_External': Path("./FigshareBrats/figshare_images"),
        # Add more datasets as needed
    }


# ============================================================================
# Dataset Classes
# ============================================================================

class ExternalBrainTumorDataset(Dataset):
    """
    Generic dataset class for external brain tumor datasets.
    Handles various directory structures and class naming conventions.
    """
    
    # Common class name mappings across different datasets
    CLASS_MAPPINGS = {
        # Kaggle original
        'glioma': 'Glioma',
        'meningioma': 'Meningioma',
        'notumor': 'No Tumor',
        'no_tumor': 'No Tumor',
        'pituitary': 'Pituitary',
        'healthy': 'No Tumor',
        'normal': 'No Tumor',
        
        # Figshare variations
        'glioma_tumor': 'Glioma',
        'meningioma_tumor': 'Meningioma',
        'pituitary_tumor': 'Pituitary',
        
        # BraTS variations  
        'hgg': 'Glioma',  # High-grade glioma
        'lgg': 'Glioma',  # Low-grade glioma
        'gbm': 'Glioma',  # Glioblastoma
    }
    
    def __init__(
        self,
        data_dir: Path,
        transform=None,
        class_filter: Optional[List[str]] = None,
        verbose: bool = True
    ):
        """
        Args:
            data_dir: Path to dataset directory
            transform: Torchvision transforms
            class_filter: Only load specific classes (default: all available)
            verbose: Print loading info
        """
        self.data_dir = Path(data_dir)
        self.transform = transform
        self.verbose = verbose
        
        self.samples = []
        self.class_to_idx = {name: idx for idx, name in enumerate(CrossValidationConfig.CLASS_NAMES)}
        self.available_classes = set()
        
        self._load_samples(class_filter)
        
    def _load_samples(self, class_filter: Optional[List[str]] = None):
        """Scan directory and load image paths with labels"""
        
        if not self.data_dir.exists():
            raise ValueError(f"Dataset directory not found: {self.data_dir}")
        
        # Scan subdirectories for class folders
        class_counts = defaultdict(int)
        
        for subdir in self.data_dir.iterdir():
            if not subdir.is_dir():
                continue
                
            # Normalize folder name to standard class
            folder_name = subdir.name.lower().strip()
            standard_class = self.CLASS_MAPPINGS.get(folder_name, None)
            
            if standard_class is None:
                if self.verbose:
                    print(f"  Warning: Unknown class folder '{folder_name}', skipping...")
                continue
            
            if class_filter and standard_class not in class_filter:
                continue
            
            self.available_classes.add(standard_class)
            label = self.class_to_idx[standard_class]
            
            # Load all image files
            for ext in ['*.jpg', '*.jpeg', '*.png', '*.gif', '*.bmp']:
                for img_path in subdir.glob(ext):
                    self.samples.append((img_path, label, standard_class))
                    class_counts[standard_class] += 1
        
        if len(self.samples) == 0:
            raise ValueError(f"No valid images found in {self.data_dir}")
        
        if self.verbose:
            print(f"\nLoaded {len(self.samples)} images from {self.data_dir.name}")
            print(f"Available classes: {sorted(self.available_classes)}")
            for cls, count in sorted(class_counts.items()):
                print(f"  - {cls}: {count} images")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label, class_name = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            # Return a blank image as fallback
            image = Image.new('RGB', (CrossValidationConfig.IMG_SIZE, CrossValidationConfig.IMG_SIZE))
        
        if self.transform:
            image = self.transform(image)
        
        return image, label
    
    def get_class_distribution(self) -> Dict[str, int]:
        """Return class distribution"""
        dist = defaultdict(int)
        for _, label, class_name in self.samples:
            dist[class_name] += 1
        return dict(dist)


class FigshareDataset(ExternalBrainTumorDataset):
    """
    Figshare Brain Tumor Dataset Handler
    
    Dataset: https://figshare.com/articles/dataset/brain_tumor_dataset/1512427
    Contains: 3,064 T1-weighted contrast-enhanced MRI images
    Classes: Glioma (1,426), Meningioma (708), Pituitary (930)
    
    NOTE: This dataset does NOT have "No Tumor" class
    """
    
    def __init__(self, data_dir: Path, transform=None, verbose: bool = True):
        # Figshare only has 3 tumor classes
        super().__init__(
            data_dir=data_dir,
            transform=transform,
            class_filter=['Glioma', 'Meningioma', 'Pituitary'],
            verbose=verbose
        )
        
        if verbose:
            print("\nNote: Figshare dataset has 3 classes (no healthy controls)")


# ============================================================================
# Evaluation Functions
# ============================================================================

def get_test_transform():
    """Get evaluation transforms (no augmentation)"""
    return T.Compose([
        T.Resize((CrossValidationConfig.IMG_SIZE, CrossValidationConfig.IMG_SIZE)),
        T.ToTensor(),
        T.Normalize(
            mean=CrossValidationConfig.NORMALIZE_MEAN,
            std=CrossValidationConfig.NORMALIZE_STD
        )
    ])


def load_pretrained_model(model_path: Path, device: str = 'cuda') -> HSANetV2:
    """Load pretrained HSANet model"""
    
    print(f"\nLoading model from {model_path}...")
    
    model = create_model(num_classes=CrossValidationConfig.NUM_CLASSES, pretrained=False)
    
    if model_path.exists():
        # Use weights_only=False for compatibility with older checkpoints
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)
        
        # Handle different checkpoint formats
        if isinstance(checkpoint, dict):
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
        else:
            state_dict = checkpoint
        
        # Remap keys if needed (handle different naming conventions)
        # The checkpoint may use 'amsm' instead of 'amsm_modules', etc.
        remapped_state_dict = {}
        for key, value in state_dict.items():
            new_key = key
            # Remap old naming convention to new
            new_key = new_key.replace('amsm.', 'amsm_modules.')
            new_key = new_key.replace('dam.', 'dam_modules.')
            remapped_state_dict[new_key] = value
        
        # Load with strict=False to handle minor mismatches
        try:
            model.load_state_dict(remapped_state_dict, strict=True)
        except RuntimeError as e:
            print(f"Strict loading failed, trying with strict=False: {e}")
            model.load_state_dict(remapped_state_dict, strict=False)
        
        print("Model loaded successfully!")
    else:
        print(f"Warning: Model file not found at {model_path}")
        print("Using randomly initialized model (for testing purposes only)")
    
    model = model.to(device)
    model.eval()
    
    return model


@torch.no_grad()
def evaluate_on_dataset(
    model: nn.Module,
    dataloader: DataLoader,
    device: str = 'cuda',
    dataset_name: str = 'External'
) -> Dict:
    """
    Evaluate model on a dataset and compute comprehensive metrics
    """
    
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_uncertainties = []
    
    print(f"\nEvaluating on {dataset_name}...")
    
    for images, labels in tqdm(dataloader, desc=f"Evaluating {dataset_name}"):
        images = images.to(device)
        
        outputs = model(images)
        
        probs = outputs['probs'].cpu().numpy()
        preds = probs.argmax(axis=1)
        uncertainties = outputs['uncertainty_total'].cpu().numpy()
        
        all_preds.extend(preds)
        all_labels.extend(labels.numpy())
        all_probs.extend(probs)
        all_uncertainties.extend(uncertainties)
    
    # Convert to numpy
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    uncertainties = np.array(all_uncertainties)
    
    # Compute metrics
    metrics = compute_metrics(y_true, y_pred, y_prob, uncertainties, dataset_name)
    
    return metrics


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    uncertainties: np.ndarray,
    dataset_name: str
) -> Dict:
    """Compute comprehensive evaluation metrics"""
    
    metrics = {
        'dataset': dataset_name,
        'total_samples': len(y_true),
    }
    
    # Find which classes are present in this dataset
    unique_classes = np.unique(y_true)
    metrics['classes_present'] = unique_classes.tolist()
    metrics['num_classes_evaluated'] = len(unique_classes)
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred) * 100
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    
    # Agreement metrics
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # Per-class metrics
    metrics['precision_per_class'] = (precision_score(y_true, y_pred, average=None, zero_division=0) * 100).tolist()
    metrics['recall_per_class'] = (recall_score(y_true, y_pred, average=None, zero_division=0) * 100).tolist()
    metrics['f1_per_class'] = (f1_score(y_true, y_pred, average=None, zero_division=0) * 100).tolist()
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    # AUC-ROC (only for classes present)
    try:
        # Filter probabilities for present classes
        y_prob_filtered = y_prob[:, unique_classes]
        metrics['auc_roc_macro'] = roc_auc_score(
            y_true, y_prob_filtered, 
            multi_class='ovr', average='macro',
            labels=unique_classes
        )
    except Exception as e:
        print(f"  Warning: AUC-ROC calculation failed: {e}")
        metrics['auc_roc_macro'] = None
    
    # ECE (Expected Calibration Error)
    metrics['ece'] = compute_ece(y_true, y_prob)
    
    # Uncertainty metrics
    metrics['uncertainty_mean'] = float(np.mean(uncertainties))
    metrics['uncertainty_std'] = float(np.std(uncertainties))
    
    correct_mask = y_true == y_pred
    metrics['uncertainty_correct'] = float(np.mean(uncertainties[correct_mask]))
    if (~correct_mask).any():
        metrics['uncertainty_incorrect'] = float(np.mean(uncertainties[~correct_mask]))
    else:
        metrics['uncertainty_incorrect'] = None
    
    # Statistical test for uncertainty separation
    if (~correct_mask).any() and correct_mask.any():
        stat, p_value = stats.mannwhitneyu(
            uncertainties[correct_mask],
            uncertainties[~correct_mask],
            alternative='less'
        )
        metrics['uncertainty_separation_pvalue'] = float(p_value)
    else:
        metrics['uncertainty_separation_pvalue'] = None
    
    return metrics


def compute_ece(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 15) -> float:
    """Compute Expected Calibration Error"""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_confidence = confidences[in_bin].mean()
            avg_accuracy = accuracies[in_bin].mean()
            ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
    
    return float(ece)


# ============================================================================
# Visualization Functions
# ============================================================================

def plot_cross_dataset_comparison(results: List[Dict], save_path: Path):
    """Create comparison plots across datasets"""
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    datasets = [r['dataset'] for r in results]
    
    # 1. Accuracy comparison
    ax1 = axes[0, 0]
    accuracies = [r['accuracy'] for r in results]
    colors = plt.cm.Set2(np.linspace(0, 1, len(datasets)))
    bars = ax1.bar(datasets, accuracies, color=colors, edgecolor='black')
    ax1.set_ylabel('Accuracy (%)', fontsize=12)
    ax1.set_title('Accuracy Across Datasets', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 105)
    for bar, acc in zip(bars, accuracies):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc:.2f}%', ha='center', va='bottom', fontsize=10)
    ax1.axhline(y=95, color='r', linestyle='--', alpha=0.5, label='95% threshold')
    ax1.legend()
    
    # 2. F1-Score comparison
    ax2 = axes[0, 1]
    f1_scores = [r['f1_macro'] for r in results]
    bars = ax2.bar(datasets, f1_scores, color=colors, edgecolor='black')
    ax2.set_ylabel('F1-Score (Macro) %', fontsize=12)
    ax2.set_title('F1-Score Across Datasets', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 105)
    for bar, f1 in zip(bars, f1_scores):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{f1:.2f}%', ha='center', va='bottom', fontsize=10)
    
    # 3. ECE comparison (calibration)
    ax3 = axes[1, 0]
    eces = [r['ece'] for r in results]
    bars = ax3.bar(datasets, eces, color=colors, edgecolor='black')
    ax3.set_ylabel('ECE (lower is better)', fontsize=12)
    ax3.set_title('Calibration (ECE) Across Datasets', fontsize=14, fontweight='bold')
    for bar, ece in zip(bars, eces):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.002,
                f'{ece:.4f}', ha='center', va='bottom', fontsize=10)
    ax3.axhline(y=0.05, color='g', linestyle='--', alpha=0.5, label='Good calibration')
    ax3.legend()
    
    # 4. Uncertainty separation
    ax4 = axes[1, 1]
    unc_correct = [r['uncertainty_correct'] for r in results]
    unc_incorrect = [r.get('uncertainty_incorrect', 0) or 0 for r in results]
    
    x = np.arange(len(datasets))
    width = 0.35
    ax4.bar(x - width/2, unc_correct, width, label='Correct predictions', color='green', alpha=0.7)
    ax4.bar(x + width/2, unc_incorrect, width, label='Incorrect predictions', color='red', alpha=0.7)
    ax4.set_xticks(x)
    ax4.set_xticklabels(datasets)
    ax4.set_ylabel('Mean Uncertainty', fontsize=12)
    ax4.set_title('Uncertainty Separation (Correct vs Incorrect)', fontsize=14, fontweight='bold')
    ax4.legend()
    
    plt.tight_layout()
    plt.savefig(save_path / 'cross_dataset_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nComparison plot saved to {save_path / 'cross_dataset_comparison.png'}")


def plot_confusion_matrices(results: List[Dict], save_path: Path):
    """Plot confusion matrices for all datasets"""
    
    n_datasets = len(results)
    fig, axes = plt.subplots(1, n_datasets, figsize=(6*n_datasets, 5))
    
    if n_datasets == 1:
        axes = [axes]
    
    for ax, result in zip(axes, results):
        cm = np.array(result['confusion_matrix'])
        
        # Get class names for present classes
        present_classes = result['classes_present']
        class_labels = [CrossValidationConfig.CLASS_NAMES[i] for i in present_classes]
        
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                   xticklabels=class_labels, yticklabels=class_labels)
        ax.set_xlabel('Predicted', fontsize=11)
        ax.set_ylabel('True', fontsize=11)
        ax.set_title(f"{result['dataset']}\nAcc: {result['accuracy']:.2f}%", fontsize=12)
    
    plt.tight_layout()
    plt.savefig(save_path / 'confusion_matrices.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Confusion matrices saved to {save_path / 'confusion_matrices.png'}")


def generate_latex_table(results: List[Dict], save_path: Path):
    """Generate LaTeX table for paper"""
    
    latex = r"""
\begin{table}[ht]
\centering
\caption{Cross-dataset validation results. HSANet trained on Kaggle Brain Tumor MRI dataset and evaluated on external datasets without any fine-tuning.}
\label{tab:cross_validation}
\begin{tabular}{lcccccc}
\toprule
\textbf{Dataset} & \textbf{Samples} & \textbf{Accuracy (\%)} & \textbf{F1 (\%)} & \textbf{AUC-ROC} & \textbf{ECE} & \textbf{$\kappa$} \\
\midrule
"""
    
    for r in results:
        auc = f"{r['auc_roc_macro']:.4f}" if r['auc_roc_macro'] else "N/A"
        latex += f"{r['dataset']} & {r['total_samples']} & {r['accuracy']:.2f} & {r['f1_macro']:.2f} & {auc} & {r['ece']:.4f} & {r['cohen_kappa']:.4f} \\\\\n"
    
    latex += r"""
\bottomrule
\end{tabular}
\end{table}
"""
    
    with open(save_path / 'cross_validation_table.tex', 'w') as f:
        f.write(latex)
    
    print(f"\nLaTeX table saved to {save_path / 'cross_validation_table.tex'}")


# ============================================================================
# Main Execution
# ============================================================================

def run_cross_dataset_validation(
    model_path: Path = None,
    datasets_config: Dict[str, Path] = None,
    output_dir: Path = None
):
    """
    Main function to run cross-dataset validation
    
    Args:
        model_path: Path to pretrained HSANet model
        datasets_config: Dictionary of dataset_name -> dataset_path
        output_dir: Directory to save results
    """
    
    # Use defaults if not provided
    model_path = model_path or CrossValidationConfig.PRETRAINED_MODEL
    output_dir = output_dir or CrossValidationConfig.OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model
    device = CrossValidationConfig.DEVICE
    model = load_pretrained_model(model_path, device)
    
    # Get transforms
    transform = get_test_transform()
    
    # Results storage
    all_results = []
    
    # Evaluate on each dataset
    if datasets_config is None:
        datasets_config = CrossValidationConfig.DATASETS
    
    for dataset_name, dataset_path in datasets_config.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"Path: {dataset_path}")
        print('='*60)
        
        if not Path(dataset_path).exists():
            print(f"  Skipping - path not found")
            continue
        
        try:
            # Create dataset
            if 'figshare' in dataset_name.lower():
                dataset = FigshareDataset(dataset_path, transform=transform)
            else:
                dataset = ExternalBrainTumorDataset(dataset_path, transform=transform)
            
            # Create dataloader
            dataloader = DataLoader(
                dataset,
                batch_size=CrossValidationConfig.BATCH_SIZE,
                shuffle=False,
                num_workers=4,
                pin_memory=True
            )
            
            # Evaluate
            metrics = evaluate_on_dataset(model, dataloader, device, dataset_name)
            all_results.append(metrics)
            
            # Print summary
            print(f"\n{dataset_name} Results:")
            print(f"  Accuracy: {metrics['accuracy']:.2f}%")
            print(f"  F1-Score: {metrics['f1_macro']:.2f}%")
            print(f"  ECE: {metrics['ece']:.4f}")
            print(f"  Cohen's Kappa: {metrics['cohen_kappa']:.4f}")
            
        except Exception as e:
            print(f"  Error evaluating {dataset_name}: {e}")
            import traceback
            traceback.print_exc()
    
    if len(all_results) == 0:
        print("\nNo datasets were successfully evaluated!")
        return
    
    # Save results
    print(f"\n{'='*60}")
    print("Saving Results...")
    print('='*60)
    
    # Save JSON results
    with open(output_dir / 'cross_validation_results.json', 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Results saved to {output_dir / 'cross_validation_results.json'}")
    
    # Generate visualizations
    plot_cross_dataset_comparison(all_results, output_dir)
    plot_confusion_matrices(all_results, output_dir)
    
    # Generate LaTeX table
    generate_latex_table(all_results, output_dir)
    
    # Print summary table
    print(f"\n{'='*60}")
    print("CROSS-DATASET VALIDATION SUMMARY")
    print('='*60)
    print(f"{'Dataset':<25} {'Accuracy':<12} {'F1':<12} {'ECE':<12}")
    print('-'*60)
    for r in all_results:
        print(f"{r['dataset']:<25} {r['accuracy']:.2f}%{'':<6} {r['f1_macro']:.2f}%{'':<6} {r['ece']:.4f}")
    print('='*60)
    
    return all_results


def download_figshare_dataset():
    """
    Instructions for downloading Figshare Brain Tumor Dataset
    """
    instructions = """
    ╔══════════════════════════════════════════════════════════════════╗
    ║           FIGSHARE BRAIN TUMOR DATASET DOWNLOAD                  ║
    ╠══════════════════════════════════════════════════════════════════╣
    ║                                                                  ║
    ║  1. Go to: https://figshare.com/articles/dataset/               ║
    ║            brain_tumor_dataset/1512427                           ║
    ║                                                                  ║
    ║  2. Download all files (3 zip files for 3 tumor types)          ║
    ║                                                                  ║
    ║  3. Extract and organize as:                                    ║
    ║     external_datasets/                                          ║
    ║       └── figshare_brain_tumor/                                 ║
    ║           ├── glioma/                                           ║
    ║           │   └── (1,426 images)                                ║
    ║           ├── meningioma/                                       ║
    ║           │   └── (708 images)                                  ║
    ║           └── pituitary/                                        ║
    ║               └── (930 images)                                  ║
    ║                                                                  ║
    ║  NOTE: This dataset does NOT have "No Tumor" class             ║
    ║        Evaluation will be on 3 classes only                     ║
    ║                                                                  ║
    ╚══════════════════════════════════════════════════════════════════╝
    """
    print(instructions)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Cross-Dataset Validation for HSANet')
    parser.add_argument('--model', type=str, default='./hsanet_results/hsanet_final.pth',
                       help='Path to pretrained model')
    parser.add_argument('--output', type=str, default='./cross_validation_results',
                       help='Output directory for results')
    parser.add_argument('--download-info', action='store_true',
                       help='Show download instructions for external datasets')
    
    args = parser.parse_args()
    
    if args.download_info:
        download_figshare_dataset()
    else:
        # Run validation
        run_cross_dataset_validation(
            model_path=Path(args.model),
            output_dir=Path(args.output)
        )
