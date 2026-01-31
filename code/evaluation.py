"""
Comprehensive Evaluation Module for HSANet
==========================================
Fixes:
1. Proper AUC-ROC calculation with probability outputs
2. Expected Calibration Error (ECE) for uncertainty validation
3. Reliability diagrams
4. Per-class and macro metrics
5. Out-of-distribution detection metrics

Author: HSANet Team
Date: January 2026
"""

import torch
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, cohen_kappa_score,
    matthews_corrcoef, roc_auc_score, roc_curve, auc,
    precision_recall_curve, average_precision_score
)
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


class MetricsCalculator:
    """Comprehensive metrics calculator with proper implementations"""
    
    def __init__(self, num_classes: int = 4, class_names: List[str] = None):
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class_{i}" for i in range(num_classes)]
        
    def compute_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        uncertainties: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Compute comprehensive classification metrics
        
        Args:
            y_true: Ground truth labels [N]
            y_pred: Predicted labels [N]
            y_prob: Predicted probabilities [N, C]
            uncertainties: Uncertainty estimates [N] (optional)
            
        Returns:
            Dictionary containing all metrics
        """
        metrics = {}
        
        # Basic metrics
        metrics['accuracy'] = accuracy_score(y_true, y_pred) * 100
        metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro') * 100
        metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro') * 100
        metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro') * 100
        metrics['precision_weighted'] = precision_score(y_true, y_pred, average='weighted') * 100
        metrics['recall_weighted'] = recall_score(y_true, y_pred, average='weighted') * 100
        metrics['f1_weighted'] = f1_score(y_true, y_pred, average='weighted') * 100
        
        # Agreement metrics
        metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
        metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
        
        # Per-class metrics
        metrics['precision_per_class'] = precision_score(y_true, y_pred, average=None).tolist()
        metrics['recall_per_class'] = recall_score(y_true, y_pred, average=None).tolist()
        metrics['f1_per_class'] = f1_score(y_true, y_pred, average=None).tolist()
        
        # Confusion matrix
        metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
        
        # Per-class support
        _, counts = np.unique(y_true, return_counts=True)
        metrics['support_per_class'] = counts.tolist()
        
        # ===== FIXED AUC-ROC CALCULATION =====
        try:
            # One-vs-Rest AUC-ROC (macro)
            metrics['auc_roc_macro'] = roc_auc_score(
                y_true, y_prob, multi_class='ovr', average='macro'
            )
            
            # One-vs-Rest AUC-ROC (weighted)
            metrics['auc_roc_weighted'] = roc_auc_score(
                y_true, y_prob, multi_class='ovr', average='weighted'
            )
            
            # Per-class AUC-ROC
            auc_per_class = []
            for i in range(self.num_classes):
                y_true_binary = (y_true == i).astype(int)
                y_prob_class = y_prob[:, i]
                if len(np.unique(y_true_binary)) > 1:
                    auc_per_class.append(roc_auc_score(y_true_binary, y_prob_class))
                else:
                    auc_per_class.append(np.nan)
            metrics['auc_roc_per_class'] = auc_per_class
            
        except Exception as e:
            print(f"Warning: AUC-ROC calculation failed: {e}")
            metrics['auc_roc_macro'] = np.nan
            metrics['auc_roc_weighted'] = np.nan
            metrics['auc_roc_per_class'] = [np.nan] * self.num_classes
        
        # Average Precision (PR-AUC)
        try:
            metrics['avg_precision_macro'] = average_precision_score(
                np.eye(self.num_classes)[y_true], y_prob, average='macro'
            )
        except:
            metrics['avg_precision_macro'] = np.nan
        
        # Calibration metrics
        if uncertainties is not None:
            metrics['uncertainty_mean'] = float(np.mean(uncertainties))
            metrics['uncertainty_std'] = float(np.std(uncertainties))
            
            # Uncertainty for correct vs incorrect predictions
            correct_mask = y_true == y_pred
            metrics['uncertainty_correct'] = float(np.mean(uncertainties[correct_mask]))
            metrics['uncertainty_incorrect'] = float(np.mean(uncertainties[~correct_mask])) if (~correct_mask).any() else np.nan
        
        return metrics
    
    def compute_ece(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 15
    ) -> Tuple[float, Dict]:
        """
        Compute Expected Calibration Error (ECE)
        
        ECE measures how well predicted probabilities align with actual accuracy.
        Lower is better. A perfectly calibrated model has ECE = 0.
        """
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        ece = 0.0
        bin_data = []
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            prop_in_bin = in_bin.mean()
            
            if prop_in_bin > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                ece += np.abs(avg_accuracy - avg_confidence) * prop_in_bin
                
                bin_data.append({
                    'bin_lower': float(bin_lower),
                    'bin_upper': float(bin_upper),
                    'avg_confidence': float(avg_confidence),
                    'avg_accuracy': float(avg_accuracy),
                    'count': int(in_bin.sum()),
                    'proportion': float(prop_in_bin)
                })
        
        return float(ece), {'bins': bin_data, 'n_bins': n_bins}
    
    def compute_mce(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray,
        n_bins: int = 15
    ) -> float:
        """Maximum Calibration Error (MCE)"""
        confidences = np.max(y_prob, axis=1)
        predictions = np.argmax(y_prob, axis=1)
        accuracies = (predictions == y_true).astype(float)
        
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers = bin_boundaries[:-1]
        bin_uppers = bin_boundaries[1:]
        
        max_gap = 0.0
        
        for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
            in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
            
            if in_bin.sum() > 0:
                avg_confidence = confidences[in_bin].mean()
                avg_accuracy = accuracies[in_bin].mean()
                gap = np.abs(avg_accuracy - avg_confidence)
                max_gap = max(max_gap, gap)
        
        return float(max_gap)
    
    def compute_roc_curves(
        self,
        y_true: np.ndarray,
        y_prob: np.ndarray
    ) -> Dict:
        """Compute ROC curves for each class"""
        roc_data = {}
        
        for i in range(self.num_classes):
            y_true_binary = (y_true == i).astype(int)
            y_prob_class = y_prob[:, i]
            
            if len(np.unique(y_true_binary)) > 1:
                fpr, tpr, thresholds = roc_curve(y_true_binary, y_prob_class)
                roc_auc = auc(fpr, tpr)
                
                roc_data[self.class_names[i]] = {
                    'fpr': fpr.tolist(),
                    'tpr': tpr.tolist(),
                    'auc': float(roc_auc)
                }
        
        return roc_data


def evaluate_model(
    model,
    dataloader,
    device: str = 'cuda',
    class_names: List[str] = None,
    save_path: Optional[Path] = None
) -> Dict:
    """
    Comprehensive model evaluation
    
    Args:
        model: Trained HSANet model
        dataloader: Test/validation dataloader
        device: Device to run evaluation on
        class_names: Names of classes
        save_path: Path to save results
        
    Returns:
        Dictionary with all metrics
    """
    model.eval()
    
    all_labels = []
    all_preds = []
    all_probs = []
    all_uncertainties = []
    all_epistemic = []
    all_aleatoric = []
    
    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            labels = labels.to(device)
            
            outputs = model(images)
            
            probs = outputs['probs']
            preds = probs.argmax(dim=1)
            
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            all_uncertainties.extend(outputs['uncertainty_total'].cpu().numpy())
            all_epistemic.extend(outputs['uncertainty_epistemic'].cpu().numpy())
            all_aleatoric.extend(outputs['uncertainty_aleatoric'].cpu().numpy())
    
    # Convert to numpy
    y_true = np.array(all_labels)
    y_pred = np.array(all_preds)
    y_prob = np.array(all_probs)
    uncertainties = np.array(all_uncertainties)
    epistemic = np.array(all_epistemic)
    aleatoric = np.array(all_aleatoric)
    
    # Calculate metrics
    num_classes = y_prob.shape[1]
    calculator = MetricsCalculator(num_classes, class_names)
    
    metrics = calculator.compute_all_metrics(y_true, y_pred, y_prob, uncertainties)
    
    # Calibration metrics
    ece, ece_data = calculator.compute_ece(y_true, y_prob)
    mce = calculator.compute_mce(y_true, y_prob)
    
    metrics['ece'] = ece
    metrics['mce'] = mce
    metrics['calibration_data'] = ece_data
    
    # ROC curves
    metrics['roc_curves'] = calculator.compute_roc_curves(y_true, y_prob)
    
    # Uncertainty breakdown
    metrics['uncertainty'] = {
        'total_mean': float(np.mean(uncertainties)),
        'total_std': float(np.std(uncertainties)),
        'epistemic_mean': float(np.mean(epistemic)),
        'epistemic_std': float(np.std(epistemic)),
        'aleatoric_mean': float(np.mean(aleatoric)),
        'aleatoric_std': float(np.std(aleatoric))
    }
    
    # Save results
    if save_path:
        save_path = Path(save_path)
        save_path.mkdir(parents=True, exist_ok=True)
        
        # Save metrics JSON
        with open(save_path / 'test_metrics.json', 'w') as f:
            # Convert non-serializable items
            metrics_json = {k: v for k, v in metrics.items() 
                          if not isinstance(v, np.ndarray)}
            json.dump(metrics_json, f, indent=2, default=str)
        
        # Save raw predictions
        np.savez(
            save_path / 'predictions.npz',
            y_true=y_true,
            y_pred=y_pred,
            y_prob=y_prob,
            uncertainties=uncertainties,
            epistemic=epistemic,
            aleatoric=aleatoric
        )
    
    return metrics


def print_evaluation_report(metrics: Dict, class_names: List[str] = None):
    """Print formatted evaluation report"""
    print("\n" + "="*60)
    print("EVALUATION REPORT")
    print("="*60)
    
    print(f"\n{'Metric':<25} {'Value':>15}")
    print("-"*45)
    print(f"{'Accuracy':<25} {metrics['accuracy']:>14.2f}%")
    print(f"{'Precision (macro)':<25} {metrics['precision_macro']:>14.2f}%")
    print(f"{'Recall (macro)':<25} {metrics['recall_macro']:>14.2f}%")
    print(f"{'F1-Score (macro)':<25} {metrics['f1_macro']:>14.2f}%")
    print(f"{'Cohen Kappa':<25} {metrics['cohen_kappa']:>14.4f}")
    print(f"{'MCC':<25} {metrics['mcc']:>14.4f}")
    print(f"{'AUC-ROC (macro)':<25} {metrics['auc_roc_macro']:>14.4f}")
    
    print(f"\n{'Calibration Metrics':}")
    print("-"*45)
    print(f"{'ECE (lower is better)':<25} {metrics['ece']:>14.4f}")
    print(f"{'MCE (lower is better)':<25} {metrics['mce']:>14.4f}")
    
    if 'uncertainty' in metrics:
        print(f"\n{'Uncertainty Metrics':}")
        print("-"*45)
        unc = metrics['uncertainty']
        print(f"{'Total Uncertainty':<25} {unc['total_mean']:>7.4f} ± {unc['total_std']:.4f}")
        print(f"{'Epistemic Uncertainty':<25} {unc['epistemic_mean']:>7.4f} ± {unc['epistemic_std']:.4f}")
        print(f"{'Aleatoric Uncertainty':<25} {unc['aleatoric_mean']:>7.4f} ± {unc['aleatoric_std']:.4f}")
    
    if class_names and 'precision_per_class' in metrics:
        print(f"\n{'Per-Class Metrics':}")
        print("-"*65)
        print(f"{'Class':<15} {'Precision':>12} {'Recall':>12} {'F1':>12} {'AUC':>12}")
        print("-"*65)
        for i, name in enumerate(class_names):
            prec = metrics['precision_per_class'][i] * 100
            rec = metrics['recall_per_class'][i] * 100
            f1 = metrics['f1_per_class'][i] * 100
            auc_val = metrics.get('auc_roc_per_class', [np.nan]*len(class_names))[i]
            print(f"{name:<15} {prec:>11.2f}% {rec:>11.2f}% {f1:>11.2f}% {auc_val:>11.4f}")
    
    print("\n" + "="*60)


if __name__ == "__main__":
    # Test with dummy data
    np.random.seed(42)
    
    n_samples = 1000
    n_classes = 4
    
    y_true = np.random.randint(0, n_classes, n_samples)
    y_prob = np.random.dirichlet(np.ones(n_classes), n_samples)
    y_pred = np.argmax(y_prob, axis=1)
    uncertainties = np.random.rand(n_samples)
    
    calculator = MetricsCalculator(n_classes, ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'])
    metrics = calculator.compute_all_metrics(y_true, y_pred, y_prob, uncertainties)
    
    ece, _ = calculator.compute_ece(y_true, y_prob)
    metrics['ece'] = ece
    metrics['mce'] = calculator.compute_mce(y_true, y_prob)
    
    print_evaluation_report(metrics, ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary'])
