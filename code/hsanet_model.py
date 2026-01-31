"""
HSANet v2: Improved Hybrid Scale-Attention Network for Brain Tumor Classification
================================================================================
Fixes and Improvements:
1. Proper Evidential Deep Learning with Dirichlet parameterization
2. Fixed uncertainty decomposition (aleatoric vs epistemic)
3. Improved multi-scale module with actual dilated convolutions
4. Better attention mechanism implementation
5. Added GradCAM support for interpretability

Author: HSANet Team
Date: January 2026
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from typing import Tuple, List, Optional, Dict
import numpy as np


class AdaptiveMultiScaleModule(nn.Module):
    """
    Adaptive Multi-Scale Module (AMSM) with Dilated Convolutions
    
    Uses parallel dilated convolutions at rates [1, 2, 4] to capture
    features at multiple receptive field scales with learned fusion weights.
    """
    def __init__(self, in_channels: int, out_channels: int = None):
        super().__init__()
        out_channels = out_channels or in_channels
        
        # Dilated convolution branches
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1, dilation=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=2, dilation=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        self.branch4 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=4, dilation=4),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
        
        # Adaptive fusion weights learned from global context
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(out_channels * 3, out_channels),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels, 3),
            nn.Softmax(dim=1)
        )
        
        # Residual projection if dimensions change
        self.residual = nn.Identity() if in_channels == out_channels else \
                        nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Multi-scale branches
        m1 = self.branch1(x)
        m2 = self.branch2(x)
        m4 = self.branch4(x)
        
        # Compute adaptive weights from concatenated global features
        concat_features = torch.cat([
            self.global_pool(m1),
            self.global_pool(m2),
            self.global_pool(m4)
        ], dim=1).flatten(1)
        
        weights = self.fc(concat_features)  # [B, 3]
        w1, w2, w4 = weights[:, 0:1, None, None], weights[:, 1:2, None, None], weights[:, 2:3, None, None]
        
        # Weighted fusion with residual
        fused = w1 * m1 + w2 * m2 + w4 * m4
        return fused + self.residual(x)


class ChannelAttention(nn.Module):
    """Channel Attention with parallel avg/max pooling (SE-Net style)"""
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, _, _ = x.shape
        avg_out = self.fc(self.avg_pool(x).view(b, c))
        max_out = self.fc(self.max_pool(x).view(b, c))
        attention = torch.sigmoid(avg_out + max_out).view(b, c, 1, 1)
        return x * attention


class SpatialAttention(nn.Module):
    """Spatial Attention with 7x7 convolution"""
    def __init__(self, kernel_size: int = 7):
        super().__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.bn = nn.BatchNorm2d(1)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        avg_out = x.mean(dim=1, keepdim=True)
        max_out = x.max(dim=1, keepdim=True)[0]
        concat = torch.cat([avg_out, max_out], dim=1)
        attention = torch.sigmoid(self.bn(self.conv(concat)))
        return x * attention


class DualAttentionModule(nn.Module):
    """
    Dual Attention Module (DAM): Sequential Channel → Spatial attention
    """
    def __init__(self, channels: int, reduction: int = 16):
        super().__init__()
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_att(x)
        x = self.spatial_att(x)
        return x


class EvidentialClassifier(nn.Module):
    """
    Evidential Deep Learning Classifier
    
    Outputs Dirichlet distribution parameters for uncertainty quantification.
    Based on Sensoy et al., "Evidential Deep Learning to Quantify Classification Uncertainty"
    """
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
        
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        logits = self.fc(x)
        
        # Evidence = softplus(logits) to ensure positivity
        evidence = F.softplus(logits)
        
        # Dirichlet parameters: alpha = evidence + 1
        alpha = evidence + 1.0
        
        # Total Dirichlet strength
        S = alpha.sum(dim=1, keepdim=True)
        
        # Expected probabilities (Dirichlet mean)
        probs = alpha / S
        
        # Uncertainty quantification
        # Total uncertainty (inversely proportional to evidence)
        uncertainty_total = self.num_classes / S.squeeze()
        
        # Aleatoric uncertainty (entropy of expected distribution)
        entropy = -torch.sum(probs * torch.log(probs + 1e-10), dim=1)
        uncertainty_aleatoric = entropy / np.log(self.num_classes)  # Normalized
        
        # Epistemic uncertainty (evidence-based)
        uncertainty_epistemic = uncertainty_total - uncertainty_aleatoric
        
        return {
            'logits': logits,
            'evidence': evidence,
            'alpha': alpha,
            'probs': probs,
            'uncertainty_total': uncertainty_total,
            'uncertainty_aleatoric': uncertainty_aleatoric,
            'uncertainty_epistemic': uncertainty_epistemic.clamp(min=0)
        }


class HSANetV2(nn.Module):
    """
    HSANet v2: Hybrid Scale-Attention Network with Evidential Learning
    
    Architecture:
    1. EfficientNet-B3 backbone (pretrained)
    2. AMSM at 3 hierarchical levels
    3. DAM for attention refinement
    4. Evidential classification head
    
    Args:
        num_classes: Number of output classes (default: 4)
        pretrained: Use ImageNet pretrained weights
        dropout: Dropout rate for classification head
    """
    def __init__(
        self,
        num_classes: int = 4,
        pretrained: bool = True,
        dropout: float = 0.3
    ):
        super().__init__()
        self.num_classes = num_classes
        
        # Backbone: EfficientNet-B3
        self.backbone = timm.create_model(
            'tf_efficientnet_b3.ns_jft_in1k',
            pretrained=pretrained,
            features_only=True,
            out_indices=[2, 3, 4]  # 3 hierarchical levels
        )
        
        # Get feature dimensions
        with torch.no_grad():
            dummy = torch.randn(1, 3, 224, 224)
            features = self.backbone(dummy)
            self.feature_dims = [f.shape[1] for f in features]
        
        # AMSM + DAM for each level
        self.amsm_modules = nn.ModuleList([
            AdaptiveMultiScaleModule(dim) for dim in self.feature_dims
        ])
        self.dam_modules = nn.ModuleList([
            DualAttentionModule(dim) for dim in self.feature_dims
        ])
        
        # Global pooling for each level
        self.pools = nn.ModuleList([
            nn.AdaptiveAvgPool2d(1) for _ in self.feature_dims
        ])
        
        # Feature fusion dimension
        total_features = sum(self.feature_dims)
        
        # Evidential classifier
        self.classifier = EvidentialClassifier(total_features, num_classes)
        
        # Store intermediate features for GradCAM
        self.gradients = None
        self.activations = None
        
    def save_gradient(self, grad):
        self.gradients = grad
        
    def forward(self, x: torch.Tensor, return_features: bool = False) -> Dict[str, torch.Tensor]:
        # Extract hierarchical features
        backbone_features = self.backbone(x)
        
        # Process through AMSM + DAM
        processed_features = []
        for i, (feat, amsm, dam, pool) in enumerate(zip(
            backbone_features, self.amsm_modules, self.dam_modules, self.pools
        )):
            ms_feat = amsm(feat)
            att_feat = dam(ms_feat)
            
            # Save last level for GradCAM
            if i == len(backbone_features) - 1:
                self.activations = att_feat
                if att_feat.requires_grad:
                    att_feat.register_hook(self.save_gradient)
            
            pooled = pool(att_feat).flatten(1)
            processed_features.append(pooled)
        
        # Concatenate all levels
        fused = torch.cat(processed_features, dim=1)
        
        # Evidential classification
        outputs = self.classifier(fused)
        
        if return_features:
            outputs['features'] = fused
            
        return outputs
    
    def predict(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Simple prediction interface returning class and uncertainty"""
        with torch.no_grad():
            outputs = self.forward(x)
            preds = outputs['probs'].argmax(dim=1)
            uncertainty = outputs['uncertainty_total']
        return preds, uncertainty


class EvidentialLoss(nn.Module):
    """
    Evidential Deep Learning Loss Function
    
    Combines:
    1. Evidence-weighted cross-entropy (type II maximum likelihood)
    2. KL divergence regularization (penalize wrong evidence)
    3. Focal loss component (class imbalance handling)
    """
    def __init__(
        self,
        num_classes: int = 4,
        lambda_kl: float = 0.2,
        lambda_focal: float = 0.3,
        annealing_epochs: int = 10,
        focal_gamma: float = 2.0
    ):
        super().__init__()
        self.num_classes = num_classes
        self.lambda_kl = lambda_kl
        self.lambda_focal = lambda_focal
        self.annealing_epochs = annealing_epochs
        self.focal_gamma = focal_gamma
        self.current_epoch = 0
        
    def set_epoch(self, epoch: int):
        self.current_epoch = epoch
        
    def forward(
        self,
        outputs: Dict[str, torch.Tensor],
        targets: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        
        alpha = outputs['alpha']
        evidence = outputs['evidence']
        probs = outputs['probs']
        
        S = alpha.sum(dim=1, keepdim=True)
        
        # One-hot encode targets
        y_onehot = F.one_hot(targets, self.num_classes).float()
        
        # 1. Evidence-weighted cross-entropy (Type II ML)
        loss_ce = torch.sum(
            y_onehot * (torch.digamma(S) - torch.digamma(alpha)),
            dim=1
        ).mean()
        
        # 2. KL divergence regularization (annealed)
        # Remove evidence from correct class before computing KL
        alpha_tilde = y_onehot + (1 - y_onehot) * alpha
        
        # KL(Dir(alpha_tilde) || Dir(1))
        sum_alpha_tilde = alpha_tilde.sum(dim=1, keepdim=True)
        kl = torch.lgamma(sum_alpha_tilde.squeeze()) - \
             torch.lgamma(torch.tensor(self.num_classes, dtype=torch.float32, device=alpha.device)) - \
             torch.sum(torch.lgamma(alpha_tilde), dim=1) + \
             torch.sum((alpha_tilde - 1) * (torch.digamma(alpha_tilde) - torch.digamma(sum_alpha_tilde)), dim=1)
        
        # Annealing coefficient
        annealing = min(1.0, self.current_epoch / self.annealing_epochs)
        loss_kl = annealing * self.lambda_kl * kl.mean()
        
        # 3. Focal loss component
        pt = torch.sum(y_onehot * probs, dim=1)
        focal_weight = (1 - pt) ** self.focal_gamma
        loss_focal = self.lambda_focal * (focal_weight * F.cross_entropy(
            outputs['logits'], targets, reduction='none'
        )).mean()
        
        # Total loss
        total_loss = loss_ce + loss_kl + loss_focal
        
        loss_dict = {
            'total': total_loss,
            'ce': loss_ce,
            'kl': loss_kl,
            'focal': loss_focal,
            'annealing': torch.tensor(annealing)
        }
        
        return total_loss, loss_dict


def create_model(num_classes: int = 4, pretrained: bool = True) -> HSANetV2:
    """Factory function to create HSANet v2 model"""
    return HSANetV2(num_classes=num_classes, pretrained=pretrained)


if __name__ == "__main__":
    # Test the model
    model = create_model()
    x = torch.randn(2, 3, 224, 224)
    
    outputs = model(x)
    print("Output keys:", outputs.keys())
    print(f"Probs shape: {outputs['probs'].shape}")
    print(f"Uncertainty total: {outputs['uncertainty_total']}")
    print(f"Uncertainty aleatoric: {outputs['uncertainty_aleatoric']}")
    print(f"Uncertainty epistemic: {outputs['uncertainty_epistemic']}")
    
    # Test loss
    criterion = EvidentialLoss()
    targets = torch.randint(0, 4, (2,))
    loss, loss_dict = criterion(outputs, targets)
    print(f"\nLoss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        print(f"  {k}: {v.item():.4f}")
    
    # Parameter count
    params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {params / 1e6:.2f}M")
