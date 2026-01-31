# HSANet: Hybrid Scale-Attention Network for Brain Tumor Classification

> **Hybrid Scale-Attention Network with Evidential Deep Learning for Uncertainty-Aware Brain Tumor Classification**

[![Accuracy](https://img.shields.io/badge/Accuracy-99.77%25-brightgreen)](.)
[![AUC-ROC](https://img.shields.io/badge/AUC--ROC-0.9999-blue)](.)
[![ECE](https://img.shields.io/badge/ECE-0.019-orange)](.)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

---

## 🏗️ Architecture

<p align="center">
  <img src="papers/CIBM/figures/architecture_diagram.png" alt="HSANet Architecture" width="800"/>
</p>

**Key Components:**
- **EfficientNet-B3 Backbone**: Pre-trained feature extraction at three hierarchical scales
- **AMSM (Adaptive Multi-Scale Module)**: Input-dependent fusion of dilated convolutions
- **DAM (Dual Attention Module)**: Channel-then-spatial attention refinement
- **Evidential Deep Learning Head**: Dirichlet-based uncertainty quantification

---

## 🎯 Key Results

| Metric | Primary Dataset | External Validation |
|--------|-----------------|---------------------|
| Samples | 1,311 | 3,064 |
| **Accuracy** | **99.77%** | **99.90%** |
| F1-Score | 99.75% | 99.88% |
| AUC-ROC | 0.9999 | - |
| ECE | 0.019 | 0.018 |
| Misclassifications | 3 | 3 |

### Performance Visualization

<p align="center">
  <img src="figures/confusion_matrix.png" alt="Confusion Matrix" width="400"/>
  <img src="figures/roc_curves.png" alt="ROC Curves" width="400"/>
</p>

### GradCAM Interpretability

<p align="center">
  <img src="figures/gradcam_grid.png" alt="GradCAM Visualization" width="700"/>
</p>

---

## 📁 Directory Structure

```
HSANet_Research/
├── 📂 code/                          # Source code
│   ├── hsanet_model.py               # HSANet architecture
│   ├── train_complete.py             # Training pipeline
│   ├── evaluation.py                 # Metrics calculation
│   ├── gradcam_visualization.py      # GradCAM interpretability
│   └── requirements.txt              # Python dependencies
│
├── 📂 notebooks/                     # Jupyter notebooks
│   ├── HSANet_Kaggle_Training.ipynb  # Main training notebook
│   └── Model_Comparison.ipynb        # ViT/Swin/ResNet comparison
│
├── 📂 models/                        # Trained model weights
│   ├── hsanet_final.pth              # Final trained model
│   └── fold_*_best.pth               # Cross-validation folds
│
├── 📂 figures/                       # Result visualizations
│   ├── confusion_matrix.png
│   ├── roc_curves.png
│   ├── reliability_diagram.png
│   └── gradcam_grid.png
│
└── 📂 papers/                        # Manuscript submissions
    └── CIBM/                         # Computers in Biology & Medicine
        ├── main.pdf                  # Compiled manuscript
        └── figures/                  # Paper figures
```

---

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/tarequejosh/HSANet-Brain-Tumor-Classification.git
cd HSANet-Brain-Tumor-Classification/code
pip install -r requirements.txt
```

### Inference

```python
from hsanet_model import HSANetV2
import torch
from PIL import Image
import torchvision.transforms as T

# Load model
model = HSANetV2(num_classes=4)
model.load_state_dict(torch.load('../models/hsanet_final.pth'))
model.eval()

# Prepare image
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open('brain_mri.jpg').convert('RGB')
input_tensor = transform(image).unsqueeze(0)

# Predict with uncertainty
with torch.no_grad():
    outputs = model(input_tensor)
    probs = outputs['probs']
    epistemic = outputs['uncertainty_epistemic']
    
classes = ['Glioma', 'Meningioma', 'No Tumor', 'Pituitary']
pred_class = classes[probs.argmax().item()]
confidence = probs.max().item()
uncertainty = epistemic.item()

print(f"Prediction: {pred_class}")
print(f"Confidence: {confidence:.2%}")
print(f"Uncertainty: {uncertainty:.4f}")
```

---

## 📊 Comparison with Other Methods

| Method | Params (M) | Accuracy (%) | Uncertainty |
|--------|------------|--------------|-------------|
| VGG-16 | 134.3 | 96.56 | ❌ |
| ResNet-50 | 23.5 | 99.12 | ❌ |
| ViT-B/16 | 86.6 | 98.94 | ❌ |
| Swin-Tiny | 28.3 | 99.21 | ❌ |
| **HSANet (Ours)** | **15.6** | **99.77** | ✅ |

*Run `notebooks/Model_Comparison.ipynb` on Kaggle to reproduce comparison experiments.*

---

## 📝 Citation

If you use this code, please cite:

```bibtex
@article{assaduzzaman2025hsanet,
  title={HSANet: Hybrid Scale-Attention Network with Evidential Deep Learning 
         for Uncertainty-Aware Brain Tumor Classification},
  author={Assaduzzaman, Md. and Josh, Md. Tareque Jamil and 
          Joy, Md. Aminur Rahman and Imti, Md. Nafish Imtiaz},
  journal={Computers in Biology and Medicine},
  year={2025},
  publisher={Elsevier}
}
```

---

## 👥 Authors

- **Md. Assaduzzaman** (Corresponding) - assaduzzaman.cse@diu.edu.bd
- **Md. Tareque Jamil Josh** - Software, Validation
- **Md. Aminur Rahman Joy** - Data Curation
- **Md. Nafish Imtiaz Imti** - Investigation

Department of Computer Science and Engineering  
Daffodil International University, Dhaka, Bangladesh

---

## 🔗 Links

- **Dataset**: [Brain Tumor MRI Dataset (Kaggle)](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **External Dataset**: [Figshare Brain Tumor Dataset](https://figshare.com/articles/dataset/brain_tumor_dataset/1512427)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
