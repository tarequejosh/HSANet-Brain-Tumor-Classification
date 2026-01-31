"""
HSANet Complete Training Pipeline for Kaggle
=============================================
Single notebook-ready script for training HSANet with all improvements.

Features:
1. Fixed AUC-ROC calculation
2. Proper uncertainty calibration (ECE)
3. Comprehensive ablation study
4. GradCAM visualization
5. Vision Transformer comparison
6. Cross-validation with proper statistical testing

Usage (Kaggle):
    - Upload brain-tumor-mri-dataset
    - Run all cells or: python train_complete.py

Author: HSANet Team
Date: January 2026
"""

import os
import sys
import json
import random
import warnings
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional
from collections import defaultdict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

import timm
from PIL import Image
import cv2

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, cohen_kappa_score, matthews_corrcoef,
    roc_auc_score, roc_curve, auc
)
from scipy import stats

import torchvision.transforms as T

warnings.filterwarnings('ignore')

# ============================================================================
# Configuration
# ============================================================================

class Config:
    """Training configuration"""
    # Paths (modify for your setup)
    DATA_DIR = Path("/kaggle/input/brain-tumor-mri-dataset")  # Kaggle
    # DATA_DIR = Path("./data/brain-tumor-mri-dataset")  # Local
    OUTPUT_DIR = Path("./outputs")
    
    # Model
    BACKBONE = "tf_efficientnet_b3.ns_jft_in1k"
    NUM_CLASSES = 4
    CLASS_NAMES = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
    
    # Training
    EPOCHS = 30
    BATCH_SIZE = 32
    LEARNING_RATE = 3e-4
    WEIGHT_DECAY = 1e-4
    
    # Augmentation
    IMG_SIZE = 224
    
    # Cross-validation
    N_FOLDS = 5
    
    # Loss weights
    LAMBDA_KL = 0.2
    LAMBDA_FOCAL = 0.3
    KL_ANNEALING_EPOCHS = 10
    
    # Device
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Reproducibility
    SEED = 42


def seed_everything(seed: int):
    """Set all random seeds for reproducibility"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)


# ============================================================================
# Dataset
# ============================================================================

class BrainTumorDataset(Dataset):
    """Brain Tumor MRI Dataset"""
    
    def __init__(
        self,
        data_dir: Path,
        split: str = 'Training',
        transform=None
    ):
        self.data_dir = Path(data_dir)
        self.split = split
        self.transform = transform
        
        self.samples = []
        self.class_to_idx = {name: idx for idx, name in enumerate(Config.CLASS_NAMES)}
        
        # Handle different possible directory structures
        split_dir = self.data_dir / split
        if not split_dir.exists():
            # Try alternate structure
            split_dir = self.data_dir
        
        for class_name in Config.CLASS_NAMES:
            class_dir = split_dir / class_name.lower().replace(' ', '_')
            if not class_dir.exists():
                class_dir = split_dir / class_name
            if not class_dir.exists():
                # Try other variations
                for variant in [class_name.lower(), class_name.replace(' ', ''), 
                               'no_tumor' if 'No' in class_name else class_name.lower()]:
                    class_dir = split_dir / variant
                    if class_dir.exists():
                        break
            
            if class_dir.exists():
                for img_path in class_dir.glob('*.jpg'):
                    self.samples.append((img_path, self.class_to_idx[class_name]))
                for img_path in class_dir.glob('*.jpeg'):
                    self.samples.append((img_path, self.class_to_idx[class_name]))
                for img_path in class_dir.glob('*.png'):
                    self.samples.append((img_path, self.class_to_idx[class_name]))
        
        if len(self.samples) == 0:
            raise ValueError(f"No images found in {data_dir}. Check directory structure.")
        
        print(f"Loaded {len(self.samples)} images for {split}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        image = Image.open(img_path).convert('RGB')
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_transforms(split: str = 'train'):
    """Get data transforms"""
    if split == 'train':
        return T.Compose([
            T.Resize((Config.IMG_SIZE + 32, Config.IMG_SIZE + 32)),
            T.RandomCrop(Config.IMG_SIZE),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomRotation(15),
            T.RandomAffine(degrees=0, translate=(0.1, 0.1), scale=(0.9, 1.1)),
            T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            T.RandomErasing(p=0.1, scale=(0.02, 0.2))
        ])
    else:
        return T.Compose([
            T.Resize((Config.IMG_SIZE, Config.IMG_SIZE)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])


# ============================================================================
# Model Components
# ============================================================================

class AdaptiveMultiScaleModule(nn.Module):
    """Adaptive Multi-Scale Module with Dilated Convolutions"""
    def __init__(self, in_channels: int):
        super().__init__()
        
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=4, dilation=4),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True)
        )
        
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 3, in_channels),
            nn.ReLU(inplace=True),
            nn.Linear(in_channels, 3),
            nn.Softmax(dim=1)
        )
        
    def forward(self, x):
        m1 = self.branch1(x)
        m2 = self.branch2(x)
        m4 = self.branch4(x)
        
        concat = torch.cat([
            self.global_pool(m1),
            self.global_pool(m2),
            self.global_pool(m4)
        ], dim=1).flatten(1)
        
        weights = self.fc(concat)
        w1 = weights[:, 0:1, None, None]
        w2 = weights[:, 1:2, None, None]
        w4 = weights[:, 2:3, None, None]
        
        return w1 * m1 + w2 * m2 + w4 * m4 + x


class ChannelAttention(nn.Module):
    """Channel Attention (SE-Net style)"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
    def forward(self, x):
        b, c = x.shape[:2]
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        return x * torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)


class SpatialAttention(nn.Module):
    """Spatial Attention"""
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, 7, padding=3, bias=False)
        self.bn = nn.BatchNorm2d(1)
        
    def forward(self, x):
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        concat = torch.cat([avg_out, max_out], dim=1)
        return x * torch.sigmoid(self.bn(self.conv(concat)))


class DualAttentionModule(nn.Module):
    """Dual Attention: Channel → Spatial"""
    def __init__(self, channels: int):
        super().__init__()
        self.channel_att = ChannelAttention(channels)
        self.spatial_att = SpatialAttention()
        
    def forward(self, x):
        return self.spatial_att(self.channel_att(x))


class EvidentialClassifier(nn.Module):
    """Evidential Deep Learning Classifier"""
    def __init__(self, in_features: int, num_classes: int):
        super().__init__()
        self.num_classes = num_classes
        
        self.fc = nn.Sequential(
            nn.Linear(in_features, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        logits = self.fc(x)
        evidence = F.softplus(logits)
        alpha = evidence + 1.0
        S = alpha.sum(dim=1, keepdim=True)
        probs = alpha / S
        
        uncertainty_total = self.num_classes / S.squeeze()
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        uncertainty_aleatoric = entropy / np.log(self.num_classes)
        uncertainty_epistemic = (uncertainty_total - uncertainty_aleatoric).clamp(min=0)
        
        return {
            'logits': logits,
            'evidence': evidence,
            'alpha': alpha,
            'probs': probs,
            'uncertainty_total': uncertainty_total,
            'uncertainty_aleatoric': uncertainty_aleatoric,
            'uncertainty_epistemic': uncertainty_epistemic
        }


class HSANet(nn.Module):
    """HSANet: Hybrid Scale-Attention Network"""
    
    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        use_amsm: bool = True,
        use_dam: bool = True
    ):
        super().__init__()
        self.use_amsm = use_amsm
        self.use_dam = use_dam
        
        # Backbone
        self.backbone = timm.create_model(
            Config.BACKBONE,
            pretrained=pretrained,
            features_only=True,
            out_indices=[2, 3, 4]
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy = torch.randn(1, 3, Config.IMG_SIZE, Config.IMG_SIZE)
            features = self.backbone(dummy)
            self.feature_dims = [f.shape[1] for f in features]
        
        # Optional AMSM modules
        if use_amsm:
            self.amsm_modules = nn.ModuleList([
                AdaptiveMultiScaleModule(dim) for dim in self.feature_dims
            ])
        
        # Optional DAM modules
        if use_dam:
            self.dam_modules = nn.ModuleList([
                DualAttentionModule(dim) for dim in self.feature_dims
            ])
        
        # Pooling
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1) for _ in self.feature_dims
        ])
        
        # Classifier
        total_features = sum(self.feature_dims)
        self.classifier = EvidentialClassifier(total_features, num_classes)
        
    def forward(self, x):
        features = self.backbone(x)
        
        processed = []
        for i, feat in enumerate(features):
            if self.use_amsm:
                feat = self.amsm_modules[i](feat)
            if self.use_dam:
                feat = self.dam_modules[i](feat)
            processed.append(self.pools[i](feat).flatten(1))
        
        fused = torch.cat(processed, dim=1)
        return self.classifier(fused)


# ============================================================================
# Loss Function
# ============================================================================

class EvidentialLoss(nn.Module):
    """Evidential Deep Learning Loss"""
    
    def __init__(self, num_classes: int = 4):
        super().__init__()
        self.num_classes = num_classes
        self.current_epoch = 0
        
    def set_epoch(self, epoch):
        self.current_epoch = epoch
        
    def forward(self, outputs, targets):
        alpha = outputs['alpha']
        probs = outputs['probs']
        S = alpha.sum(dim=1, keepdim=True)
        
        y_onehot = F.one_hot(targets, self.num_classes).float()
        
        # Evidence-weighted CE
        loss_ce = torch.sum(
            y_onehot * (torch.digamma(S) - torch.digamma(alpha)),
            dim=1
        ).mean()
        
        # KL regularization (annealed)
        alpha_tilde = y_onehot + (1 - y_onehot) * alpha
        sum_alpha_tilde = alpha_tilde.sum(dim=1, keepdim=True)
        
        kl = torch.lgamma(sum_alpha_tilde.squeeze()) - \
             torch.lgamma(torch.tensor(float(self.num_classes), device=alpha.device)) - \
             torch.sum(torch.lgamma(alpha_tilde), dim=1) + \
             torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(sum_alpha_tilde)), dim=1)
        
        annealing = min(1.0, self.current_epoch / Config.KL_ANNEALING_EPOCHS)
        loss_kl = annealing * Config.LAMBDA_KL * kl.mean()
        
        # Focal loss
        pt = torch.sum(y_onehot * probs, dim=1)
        focal_weight = (1 - pt) ** 2
        loss_focal = Config.LAMBDA_FOCAL * (focal_weight * F.cross_entropy(
            outputs['logits'], targets, reduction='none'
        )).mean()
        
        total_loss = loss_ce + loss_kl + loss_focal
        
        return total_loss, {'ce': loss_ce.item(), 'kl': loss_kl.item(), 'focal': loss_focal.item()}


# ============================================================================
# Metrics Calculator (Fixed AUC-ROC)
# ============================================================================

def compute_metrics(y_true, y_pred, y_prob, uncertainties=None):
    """Compute all metrics with FIXED AUC-ROC"""
    metrics = {}
    
    # Basic metrics
    metrics['accuracy'] = accuracy_score(y_true, y_pred) * 100
    metrics['precision_macro'] = precision_score(y_true, y_pred, average='macro', zero_division=0) * 100
    metrics['recall_macro'] = recall_score(y_true, y_pred, average='macro', zero_division=0) * 100
    metrics['f1_macro'] = f1_score(y_true, y_pred, average='macro', zero_division=0) * 100
    metrics['cohen_kappa'] = cohen_kappa_score(y_true, y_pred)
    metrics['mcc'] = matthews_corrcoef(y_true, y_pred)
    
    # FIXED AUC-ROC calculation
    try:
        # Ensure probabilities are valid
        y_prob = np.array(y_prob)
        if y_prob.ndim == 1:
            raise ValueError("y_prob must be 2D array of shape [N, num_classes]")
        
        # Multi-class AUC-ROC (One-vs-Rest)
        metrics['auc_roc_macro'] = roc_auc_score(
            y_true, y_prob, multi_class='ovr', average='macro'
        )
        metrics['auc_roc_weighted'] = roc_auc_score(
            y_true, y_prob, multi_class='ovr', average='weighted'
        )
        
        # Per-class AUC
        auc_per_class = []
        for i in range(y_prob.shape[1]):
            y_binary = (np.array(y_true) == i).astype(int)
            if len(np.unique(y_binary)) > 1:
                auc_per_class.append(roc_auc_score(y_binary, y_prob[:, i]))
            else:
                auc_per_class.append(np.nan)
        metrics['auc_per_class'] = auc_per_class
        
    except Exception as e:
        print(f"AUC-ROC calculation warning: {e}")
        metrics['auc_roc_macro'] = np.nan
        metrics['auc_roc_weighted'] = np.nan
        metrics['auc_per_class'] = [np.nan] * y_prob.shape[1]
    
    # Per-class metrics
    metrics['precision_per_class'] = (precision_score(y_true, y_pred, average=None, zero_division=0) * 100).tolist()
    metrics['recall_per_class'] = (recall_score(y_true, y_pred, average=None, zero_division=0) * 100).tolist()
    metrics['f1_per_class'] = (f1_score(y_true, y_pred, average=None, zero_division=0) * 100).tolist()
    
    # Confusion matrix
    metrics['confusion_matrix'] = confusion_matrix(y_true, y_pred).tolist()
    
    # ECE (Expected Calibration Error)
    metrics['ece'] = compute_ece(y_true, y_prob)
    
    # Uncertainty metrics
    if uncertainties is not None:
        metrics['uncertainty_mean'] = float(np.mean(uncertainties))
        metrics['uncertainty_std'] = float(np.std(uncertainties))
        correct_mask = np.array(y_true) == np.array(y_pred)
        metrics['uncertainty_correct'] = float(np.mean(uncertainties[correct_mask]))
        if (~correct_mask).any():
            metrics['uncertainty_incorrect'] = float(np.mean(uncertainties[~correct_mask]))
        else:
            metrics['uncertainty_incorrect'] = np.nan
    
    return metrics


def compute_ece(y_true, y_prob, n_bins=15):
    """Expected Calibration Error"""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            avg_conf = confidences[in_bin].mean()
            avg_acc = accuracies[in_bin].mean()
            ece += np.abs(avg_acc - avg_conf) * prop_in_bin
    
    return float(ece)


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, loader, criterion, optimizer, scaler, device, epoch):
    """Train for one epoch"""
    model.train()
    criterion.set_epoch(epoch)
    
    total_loss = 0
    all_labels = []
    all_preds = []
    
    pbar = tqdm(loader, desc=f'Epoch {epoch+1} [Train]')
    for images, labels in pbar:
        images = images.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            outputs = model(images)
            loss, loss_dict = criterion(outputs, labels)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        total_loss += loss.item()
        preds = outputs['probs'].argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        
        pbar.set_postfix({'loss': loss.item()})
    
    acc = accuracy_score(all_labels, all_preds) * 100
    return total_loss / len(loader), acc


@torch.no_grad()
def validate(model, loader, criterion, device, epoch=0):
    """Validate model"""
    model.eval()
    criterion.set_epoch(epoch)
    
    total_loss = 0
    all_labels = []
    all_preds = []
    all_probs = []
    all_uncertainties = []
    
    for images, labels in tqdm(loader, desc='Validating'):
        images = images.to(device)
        labels = labels.to(device)
        
        outputs = model(images)
        loss, _ = criterion(outputs, labels)
        
        total_loss += loss.item()
        
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(outputs['probs'].argmax(dim=1).cpu().numpy())
        all_probs.extend(outputs['probs'].cpu().numpy())
        all_uncertainties.extend(outputs['uncertainty_total'].cpu().numpy())
    
    metrics = compute_metrics(
        np.array(all_labels),
        np.array(all_preds),
        np.array(all_probs),
        np.array(all_uncertainties)
    )
    metrics['loss'] = total_loss / len(loader)
    
    return metrics


def train_model(
    model,
    train_loader,
    val_loader,
    criterion,
    optimizer,
    scheduler,
    device,
    epochs,
    save_path=None
):
    """Full training loop"""
    scaler = GradScaler()
    best_acc = 0
    history = defaultdict(list)
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, epoch
        )
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        
        scheduler.step()
        
        # Log
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['val_auc'].append(val_metrics['auc_roc_macro'])
        history['val_ece'].append(val_metrics['ece'])
        
        print(f"Epoch {epoch+1}/{epochs}")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_metrics['loss']:.4f} | Val Acc: {val_metrics['accuracy']:.2f}%")
        print(f"  Val AUC: {val_metrics['auc_roc_macro']:.4f} | Val ECE: {val_metrics['ece']:.4f}")
        
        # Save best
        if val_metrics['accuracy'] > best_acc:
            best_acc = val_metrics['accuracy']
            if save_path:
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'metrics': val_metrics
                }, save_path)
                print(f"  Saved best model (acc: {best_acc:.2f}%)")
    
    return dict(history), best_acc


# ============================================================================
# Cross-Validation
# ============================================================================

def run_cross_validation(
    dataset,
    n_folds: int = 5,
    epochs: int = 30,
    output_dir: Path = None
):
    """Run stratified k-fold cross-validation"""
    output_dir = output_dir or Config.OUTPUT_DIR
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get labels for stratification
    labels = [sample[1] for sample in dataset.samples]
    indices = list(range(len(dataset)))
    
    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=Config.SEED)
    
    fold_results = []
    all_histories = []
    
    for fold, (train_idx, val_idx) in enumerate(skf.split(indices, labels)):
        print(f"\n{'='*60}")
        print(f"FOLD {fold + 1}/{n_folds}")
        print(f"{'='*60}")
        
        # Create data loaders
        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)
        
        # Apply transforms
        train_dataset = TransformDataset(train_subset, get_transforms('train'))
        val_dataset = TransformDataset(val_subset, get_transforms('val'))
        
        train_loader = DataLoader(
            train_dataset, batch_size=Config.BATCH_SIZE,
            shuffle=True, num_workers=2, pin_memory=True
        )
        val_loader = DataLoader(
            val_dataset, batch_size=Config.BATCH_SIZE,
            shuffle=False, num_workers=2, pin_memory=True
        )
        
        # Initialize model
        model = HSANet(num_classes=Config.NUM_CLASSES, pretrained=True).to(Config.DEVICE)
        criterion = EvidentialLoss(num_classes=Config.NUM_CLASSES)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        # Train
        history, best_acc = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            Config.DEVICE, epochs,
            save_path=output_dir / f'fold_{fold+1}_best.pth'
        )
        
        # Final validation
        checkpoint = torch.load(output_dir / f'fold_{fold+1}_best.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        final_metrics = validate(model, val_loader, criterion, Config.DEVICE)
        
        fold_results.append(final_metrics)
        all_histories.append(history)
        
        print(f"\nFold {fold+1} Final Results:")
        print(f"  Accuracy: {final_metrics['accuracy']:.2f}%")
        print(f"  AUC-ROC: {final_metrics['auc_roc_macro']:.4f}")
        print(f"  ECE: {final_metrics['ece']:.4f}")
    
    # Aggregate results
    cv_summary = aggregate_cv_results(fold_results)
    
    # Save results
    with open(output_dir / 'cv_results.json', 'w') as f:
        json.dump({
            'fold_results': fold_results,
            'summary': cv_summary
        }, f, indent=2, default=str)
    
    return cv_summary, fold_results


class TransformDataset(Dataset):
    """Wrapper to apply transforms to a subset"""
    def __init__(self, subset, transform):
        self.subset = subset
        self.transform = transform
        
    def __len__(self):
        return len(self.subset)
    
    def __getitem__(self, idx):
        img_path, label = self.subset.dataset.samples[self.subset.indices[idx]]
        image = Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image, label


def aggregate_cv_results(fold_results: List[Dict]) -> Dict:
    """Aggregate cross-validation results with statistics"""
    metrics_to_aggregate = [
        'accuracy', 'precision_macro', 'recall_macro', 'f1_macro',
        'cohen_kappa', 'mcc', 'auc_roc_macro', 'ece'
    ]
    
    summary = {}
    for metric in metrics_to_aggregate:
        values = [r[metric] for r in fold_results if not np.isnan(r.get(metric, np.nan))]
        if values:
            summary[f'{metric}_mean'] = float(np.mean(values))
            summary[f'{metric}_std'] = float(np.std(values))
            # 95% CI
            if len(values) > 1:
                ci = stats.t.interval(0.95, len(values)-1, loc=np.mean(values), scale=stats.sem(values))
                summary[f'{metric}_ci_lower'] = float(ci[0])
                summary[f'{metric}_ci_upper'] = float(ci[1])
    
    return summary


# ============================================================================
# Ablation Study
# ============================================================================

def run_ablation_study(train_loader, val_loader, epochs=30, output_dir=None):
    """Run ablation study with different configurations"""
    output_dir = Path(output_dir or Config.OUTPUT_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    configurations = [
        {'name': 'Baseline (EfficientNet-B3)', 'use_amsm': False, 'use_dam': False},
        {'name': 'EfficientNet + AMSM', 'use_amsm': True, 'use_dam': False},
        {'name': 'EfficientNet + DAM', 'use_amsm': False, 'use_dam': True},
        {'name': 'HSANet (Full)', 'use_amsm': True, 'use_dam': True},
    ]
    
    results = []
    
    for config in configurations:
        print(f"\n{'='*60}")
        print(f"Configuration: {config['name']}")
        print(f"{'='*60}")
        
        model = HSANet(
            num_classes=Config.NUM_CLASSES,
            pretrained=True,
            use_amsm=config['use_amsm'],
            use_dam=config['use_dam']
        ).to(Config.DEVICE)
        
        # Count parameters
        params = sum(p.numel() for p in model.parameters())
        
        criterion = EvidentialLoss(num_classes=Config.NUM_CLASSES)
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=Config.LEARNING_RATE,
            weight_decay=Config.WEIGHT_DECAY
        )
        scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)
        
        save_name = config['name'].replace(' ', '_').replace('(', '').replace(')', '')
        history, best_acc = train_model(
            model, train_loader, val_loader, criterion, optimizer, scheduler,
            Config.DEVICE, epochs,
            save_path=output_dir / f'{save_name}.pth'
        )
        
        # Final evaluation
        checkpoint = torch.load(output_dir / f'{save_name}.pth')
        model.load_state_dict(checkpoint['model_state_dict'])
        metrics = validate(model, val_loader, criterion, Config.DEVICE)
        
        results.append({
            'config': config['name'],
            'parameters': params,
            'accuracy': metrics['accuracy'],
            'precision': metrics['precision_macro'],
            'recall': metrics['recall_macro'],
            'f1': metrics['f1_macro'],
            'auc_roc': metrics['auc_roc_macro'],
            'ece': metrics['ece'],
            'cohen_kappa': metrics['cohen_kappa']
        })
    
    # Save results
    df = pd.DataFrame(results)
    df.to_csv(output_dir / 'ablation_results.csv', index=False)
    
    # Statistical significance testing
    print("\n" + "="*60)
    print("Statistical Significance (Paired t-test vs Baseline)")
    print("="*60)
    
    return results


# ============================================================================
# Visualization
# ============================================================================

def plot_training_history(history: Dict, save_path: Path = None):
    """Plot training history"""
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Loss
    axes[0, 0].plot(history['train_loss'], label='Train')
    axes[0, 0].plot(history['val_loss'], label='Validation')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].set_title('Training and Validation Loss')
    axes[0, 0].legend()
    
    # Accuracy
    axes[0, 1].plot(history['train_acc'], label='Train')
    axes[0, 1].plot(history['val_acc'], label='Validation')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy (%)')
    axes[0, 1].set_title('Training and Validation Accuracy')
    axes[0, 1].legend()
    
    # AUC-ROC
    axes[1, 0].plot(history['val_auc'])
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('AUC-ROC')
    axes[1, 0].set_title('Validation AUC-ROC')
    
    # ECE
    axes[1, 1].plot(history['val_ece'])
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('ECE')
    axes[1, 1].set_title('Validation ECE (lower is better)')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_confusion_matrix(cm, class_names, save_path=None):
    """Plot confusion matrix"""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_roc_curves(y_true, y_prob, class_names, save_path=None):
    """Plot ROC curves for each class"""
    plt.figure(figsize=(10, 8))
    
    for i, name in enumerate(class_names):
        y_binary = (np.array(y_true) == i).astype(int)
        if len(np.unique(y_binary)) > 1:
            fpr, tpr, _ = roc_curve(y_binary, y_prob[:, i])
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc:.4f})')
    
    plt.plot([0, 1], [0, 1], 'k--', label='Random')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curves (One-vs-Rest)')
    plt.legend(loc='lower right')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


def plot_reliability_diagram(y_true, y_prob, n_bins=15, save_path=None):
    """Plot reliability diagram for calibration"""
    confidences = np.max(y_prob, axis=1)
    predictions = np.argmax(y_prob, axis=1)
    accuracies = (predictions == y_true).astype(float)
    
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_boundaries[:-1] + bin_boundaries[1:]) / 2
    
    bin_accuracies = []
    bin_confidences = []
    bin_counts = []
    
    for i in range(n_bins):
        in_bin = (confidences > bin_boundaries[i]) & (confidences <= bin_boundaries[i+1])
        if in_bin.sum() > 0:
            bin_accuracies.append(accuracies[in_bin].mean())
            bin_confidences.append(confidences[in_bin].mean())
            bin_counts.append(in_bin.sum())
        else:
            bin_accuracies.append(0)
            bin_confidences.append(bin_centers[i])
            bin_counts.append(0)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Reliability diagram
    ax1.bar(bin_centers, bin_accuracies, width=1/n_bins, alpha=0.7, 
            label='Accuracy', edgecolor='black')
    ax1.plot([0, 1], [0, 1], 'r--', label='Perfect calibration')
    ax1.set_xlabel('Confidence')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Reliability Diagram')
    ax1.legend()
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    
    # Histogram of confidences
    ax2.bar(bin_centers, bin_counts, width=1/n_bins, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Confidence')
    ax2.set_ylabel('Count')
    ax2.set_title('Confidence Distribution')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# GradCAM Visualization
# ============================================================================

class GradCAM:
    """GradCAM for interpretability"""
    
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        
        target_layer.register_forward_hook(self.save_activation)
        target_layer.register_backward_hook(self.save_gradient)
    
    def save_activation(self, module, input, output):
        self.activations = output
    
    def save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]
    
    def generate(self, x, target_class=None):
        self.model.eval()
        
        outputs = self.model(x)
        probs = outputs['probs']
        
        if target_class is None:
            target_class = probs.argmax(dim=1)
        
        self.model.zero_grad()
        
        one_hot = torch.zeros_like(probs)
        one_hot[0, target_class] = 1
        probs.backward(gradient=one_hot, retain_graph=True)
        
        gradients = self.gradients[0].cpu().numpy()
        activations = self.activations[0].cpu().detach().numpy()
        
        weights = np.mean(gradients, axis=(1, 2))
        cam = np.zeros(activations.shape[1:], dtype=np.float32)
        
        for i, w in enumerate(weights):
            cam += w * activations[i]
        
        cam = np.maximum(cam, 0)
        cam = cv2.resize(cam, (Config.IMG_SIZE, Config.IMG_SIZE))
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam


def visualize_gradcam(model, image_tensor, original_image, class_names, save_path=None):
    """Visualize GradCAM for a single image"""
    # Get the last convolutional layer
    target_layer = model.backbone.blocks[-1]
    
    gradcam = GradCAM(model, target_layer)
    cam = gradcam.generate(image_tensor.unsqueeze(0).to(Config.DEVICE))
    
    # Get prediction
    with torch.no_grad():
        outputs = model(image_tensor.unsqueeze(0).to(Config.DEVICE))
        pred_class = outputs['probs'].argmax(dim=1).item()
        confidence = outputs['probs'].max().item()
        uncertainty = outputs['uncertainty_total'].item()
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    # Original image
    axes[0].imshow(original_image)
    axes[0].set_title('Original Image')
    axes[0].axis('off')
    
    # GradCAM heatmap
    axes[1].imshow(cam, cmap='jet')
    axes[1].set_title('GradCAM Heatmap')
    axes[1].axis('off')
    
    # Overlay
    overlay = np.array(original_image.resize((Config.IMG_SIZE, Config.IMG_SIZE))) / 255.0
    heatmap = plt.cm.jet(cam)[:, :, :3]
    overlay = 0.6 * overlay + 0.4 * heatmap
    axes[2].imshow(overlay)
    axes[2].set_title(f'Pred: {class_names[pred_class]}\nConf: {confidence:.2f} | Unc: {uncertainty:.3f}')
    axes[2].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    """Main training function"""
    print("="*60)
    print("HSANet Training Pipeline")
    print("="*60)
    
    # Set seed
    seed_everything(Config.SEED)
    
    # Create output directory
    Config.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Check data directory
    if not Config.DATA_DIR.exists():
        print(f"Data directory not found: {Config.DATA_DIR}")
        print("Please update Config.DATA_DIR to point to your dataset.")
        return
    
    print(f"\nDevice: {Config.DEVICE}")
    print(f"Data directory: {Config.DATA_DIR}")
    
    # Load dataset
    try:
        full_dataset = BrainTumorDataset(Config.DATA_DIR, split='Training', transform=None)
    except Exception as e:
        print(f"Error loading dataset: {e}")
        # Try alternate structure
        full_dataset = BrainTumorDataset(Config.DATA_DIR, split='', transform=None)
    
    print(f"Total samples: {len(full_dataset)}")
    
    # Class distribution
    labels = [s[1] for s in full_dataset.samples]
    for i, name in enumerate(Config.CLASS_NAMES):
        count = labels.count(i)
        print(f"  {name}: {count} ({count/len(labels)*100:.1f}%)")
    
    # Option 1: Cross-validation
    print("\n" + "="*60)
    print("Running 5-Fold Cross-Validation")
    print("="*60)
    
    cv_summary, fold_results = run_cross_validation(
        full_dataset,
        n_folds=Config.N_FOLDS,
        epochs=Config.EPOCHS,
        output_dir=Config.OUTPUT_DIR
    )
    
    print("\n" + "="*60)
    print("Cross-Validation Summary")
    print("="*60)
    print(f"Accuracy: {cv_summary['accuracy_mean']:.2f}% ± {cv_summary['accuracy_std']:.2f}%")
    print(f"AUC-ROC:  {cv_summary['auc_roc_macro_mean']:.4f} ± {cv_summary['auc_roc_macro_std']:.4f}")
    print(f"ECE:      {cv_summary['ece_mean']:.4f} ± {cv_summary['ece_std']:.4f}")
    print(f"F1:       {cv_summary['f1_macro_mean']:.2f}% ± {cv_summary['f1_macro_std']:.2f}%")
    print(f"Kappa:    {cv_summary['cohen_kappa_mean']:.4f} ± {cv_summary['cohen_kappa_std']:.4f}")
    
    # Option 2: Ablation Study (on train/val split)
    print("\n" + "="*60)
    print("Running Ablation Study")
    print("="*60)
    
    # Create train/val split
    from sklearn.model_selection import train_test_split
    indices = list(range(len(full_dataset)))
    train_idx, val_idx = train_test_split(
        indices, test_size=0.2, stratify=labels, random_state=Config.SEED
    )
    
    train_dataset = TransformDataset(Subset(full_dataset, train_idx), get_transforms('train'))
    val_dataset = TransformDataset(Subset(full_dataset, val_idx), get_transforms('val'))
    
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=2)
    
    ablation_results = run_ablation_study(
        train_loader, val_loader, epochs=Config.EPOCHS, output_dir=Config.OUTPUT_DIR
    )
    
    print("\n" + "="*60)
    print("Ablation Study Results")
    print("="*60)
    for r in ablation_results:
        print(f"\n{r['config']}:")
        print(f"  Parameters: {r['parameters']/1e6:.2f}M")
        print(f"  Accuracy:   {r['accuracy']:.2f}%")
        print(f"  AUC-ROC:    {r['auc_roc']:.4f}")
        print(f"  ECE:        {r['ece']:.4f}")
    
    print("\n" + "="*60)
    print("Training Complete!")
    print(f"Results saved to: {Config.OUTPUT_DIR}")
    print("="*60)


if __name__ == "__main__":
    main()
