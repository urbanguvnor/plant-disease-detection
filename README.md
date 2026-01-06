# Plant Disease Detection Using Deep Learning Group 7

A deep learning system for automated plant disease detection using ResNet-50 with transfer learning, achieving **99.85% test accuracy** on 29 plant disease categories.

---

##  Project Team

**Institution:** Bells University of Technology, Ota, Ogun State, Nigeria  
**Department:** Information & Communication Technology  
**Course:** ICT423 - Machine Learning/Deep Learning (400 Level)  
**Supervisor:** Ayuba Muhammad, New Horizons ICT  
**Session:** 2024/2025

| S/N | Name | Matric No. | Email | Role |
|-----|------|------------|-------|------|
| 1 | Osemudiame Okoeguale | 2022/12013 | iamose@aol.com | Team Lead & Implementation |
| 2 | Ezenwoko Kamsi Enyinnaya | 2022/11558 | Kamsiezenwoko@gmail.com | Model Training & Optimization |
| 3 | Fabiawari Ryan Douglas | 2022/11782 | fabiadouglasm75@aol.com | Evaluation & Analysis |
| 4 | Odighizuwa Steven | 2022/11641 | odighizuwa2.9@gmail.com | Literature Review |
| 5 | Ikechukwu Emmanuel Okechukwu | 2023/12807 | emma.ik2na@gmail.com | Documentation & Reporting |
| 6 | Emmanuel Destiny Awereka | 2022/11310 | Awesomereka@gmail.com | Documentation & Reporting |

---

##  Table of Contents

1. [Project Overview](#project-overview)
2. [Key Results](#key-results)
3. [Methodology](#methodology)
4. [Installation & Setup](#installation--setup)
5. [Usage Instructions](#usage-instructions)
6. [Results & Discussion](#results--discussion)
7. [Conclusion](#conclusion)


---

##  Project Overview

### Problem Statement

Plant diseases cause 20-40% annual crop losses globally, threatening food security and farmers' livelihoods. Traditional manual inspection is time-consuming, requires expertise, and is often unavailable to small-scale Nigerian farmers.

### Solution

We developed an AI-powered plant disease detection system using deep learning that can identify 29 different plant diseases from leaf images with 99.85% accuracy, providing a fast, accurate, and accessible diagnostic tool.

### Objectives

1. Implement ResNet-50 CNN for plant disease classification
2. Achieve >90% accuracy on multi-class disease detection
3. Evaluate model performance using standard metrics
4. Document findings in a publication-ready research paper

### Significance

- **For Farmers:** Instant disease diagnosis via smartphone
- **For Agriculture:** Reduced crop losses through early detection
- **For Nigeria:** Improved food security and agricultural productivity
- **For Academia:** Demonstrates practical ML application in agriculture

---

##  Key Results

### Overall Performance

| Metric | Score |
|--------|-------|
| **Test Accuracy** | **99.85%** |
| **Precision** | **99.86%** |
| **Recall** | **99.85%** |
| **F1-Score** | **99.85%** |
| **Validation Accuracy** | **99.75%** |

### Dataset Statistics

- **Total Images:** 67,111
- **Training Set:** 53,690 (80%)
- **Validation Set:** 12,067 (18%)
- **Test Set:** 1,354 (2%)
- **Classes:** 29 plant diseases
- **Plant Species:** 9 (Apple, Bell Pepper, Cherry, Corn, Grape, Peach, Potato, Strawberry, Tomato)

### Key Achievements

 **27 out of 29 classes** achieved 100% test accuracy  
 Only **2 misclassifications** out of 1,354 test images  
 Training completed in **~1.5 hours** on RTX 4060 GPU  
 **15 epochs** sufficient for convergence  
 **23.6M parameters** fully optimized  

---

##  Methodology

### 1. Model Architecture

**Model:** ResNet-50 (Residual Network with 50 layers)  
**Technique:** Transfer Learning with ImageNet pre-trained weights  
**Input Size:** 224×224×3 RGB images  
**Output:** 29-class softmax classification  
**Parameters:** 23,567,453 trainable parameters  

**Why ResNet-50?**
- Proven effectiveness in image classification
- Residual connections prevent vanishing gradients
- Pre-trained on ImageNet provides strong feature extraction
- Suitable for agricultural image analysis

### 2. Data Preprocessing

**Image Transformations:**
- Resize to 224×224 pixels
- Normalization (ImageNet statistics)
- Data augmentation: horizontal flipping, rotation (±10°), color jittering

**Dataset Split:**
- Training: 80% (with augmentation)
- Validation: 18% (no augmentation)
- Test: 2% (no augmentation)

### 3. Training Configuration

| Parameter | Value |
|-----------|-------|
| Optimizer | Adam |
| Learning Rate | 0.001 (initial) |
| LR Scheduler | ReduceLROnPlateau |
| Batch Size | 32 |
| Epochs | 15 |
| Loss Function | Cross-Entropy |
| Hardware | NVIDIA RTX 4060 (8GB VRAM) |
| Framework | PyTorch 2.9.1 with CUDA 12.4 |

### 4. Evaluation Metrics

- **Accuracy:** Overall correctness
- **Precision:** True positive rate
- **Recall:** Sensitivity to all positive cases
- **F1-Score:** Harmonic mean of precision and recall
- **Confusion Matrix:** Class-wise performance visualization

---

##  Installation & Setup

### System Requirements

- Python 3.10+
- NVIDIA GPU with CUDA support (recommended)
- 8GB RAM minimum
- 10GB free disk space

### Installation Steps

```bash
# 1. Clone repository
git clone https://github.com/YOUR_USERNAME/plant-disease-detection.git
cd plant-disease-detection

# 2. Install PyTorch with CUDA
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# 3. Install dependencies
pip install numpy pandas scikit-learn Pillow opencv-python matplotlib seaborn tqdm

# 4. Verify GPU
python -c "import torch; print('CUDA:', torch.cuda.is_available())"
```

### Project Structure

```
plant-disease-detection/
├── data/                          # Dataset (67,111 images)
├── data_preprocessing_custom.py   # Data loading
├── model.py                       # ResNet-50 architecture
├── train.py                       # Training script
├── evaluate.py                    # Evaluation script
├── models/best_model.pth          # Trained model (90MB)
└── results/                       # Outputs and visualizations
```

---

##  Usage Instructions

### Training the Model

```bash
python train.py
```

**Output:** Trained model saved to `models/best_model.pth`

### Evaluating the Model

```bash
python evaluate.py
```

**Output:** 
- Test metrics in `results/test_metrics.json`
- Confusion matrix: `results/confusion_matrix.png`
- Per-class accuracy: `results/per_class_accuracy.png`
- Classification report: `results/classification_report.txt`

### Making Predictions

```python
import torch
from PIL import Image
from torchvision import transforms
from model import create_model

# Load model
model = create_model(num_classes=29, model_type='resnet50', pretrained=False)
checkpoint = torch.load('models/best_model.pth')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

# Prepare image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

image = Image.open('leaf_image.jpg')
input_tensor = transform(image).unsqueeze(0)

# Predict
with torch.no_grad():
    output = model(input_tensor)
    _, predicted = torch.max(output, 1)
    print(f"Predicted disease class: {predicted.item()}")
```

---

##  Results & Discussion

### Quantitative Results

**Overall Test Performance:**
- Accuracy: 99.85% (1352/1354 correct predictions)
- Precision: 99.86%
- Recall: 99.85%
- F1-Score: 99.85%

**Perfect Classification (100% Accuracy) on 27 Classes:**

All diseases in the following categories achieved perfect scores:
- **Tomato** (5 diseases): All 100%
- **Apple** (4 diseases): All 100%
- **Grape** (4 diseases): All 100%
- **Potato** (3 diseases): All 100%
- **Bell Pepper** (2 conditions): All 100%
- **Cherry** (2 conditions): All 100%
- **Peach** (2 conditions): All 100%
- **Strawberry** (2 conditions): All 100%

**Classes with Minor Errors:**
- Corn Northern Leaf Blight: 95.83% (2 misclassifications)
- Corn Cercospora Leaf Spot: 97.62% (high precision despite visual similarity)

### Error Analysis

**Total Errors:** 2 out of 1,354 test images (0.15% error rate)

**Error Pattern:**
Both misclassifications occurred between visually similar corn diseases (Cercospora Leaf Spot ↔ Northern Leaf Blight), which share overlapping early-stage symptoms.

**Model Strengths:**
- Zero confusion between different plant species
- Perfect recognition of all non-corn diseases
- Excellent generalization (test > validation accuracy)
- Robust to image variations

### Comparison with Literature

Our model's 99.85% accuracy compares favorably with recent studies:
- Mohanty et al. (2016): 99.35% on PlantVillage
- Ferentinos (2018): 99.53% on combined datasets
- **Our work: 99.85%** on 29-class multi-species dataset

### Training Efficiency

- Converged in 15 epochs (~1.5 hours)
- No overfitting observed (train/val gap < 0.1%)
- Efficient resource utilization on mid-range GPU
- Suitable for resource-constrained environments

### Practical Implications

**For Nigerian Agriculture:**
- Deployable on smartphones for field use
- No internet required (on-device inference)
- Supports multiple crop types common in Nigeria
- Cost-effective alternative to expert consultations

**Deployment Considerations:**
- Model size: 90MB (suitable for mobile apps)
- Inference time: ~25ms per image
- Works offline after initial download
- Can be integrated with extension services

---

## Conclusion

This project successfully developed a deep learning system for plant disease detection achieving 99.85% test accuracy. The ResNet-50 model with transfer learning proved highly effective, correctly classifying 1,352 out of 1,354 test images across 29 disease categories.

**Key Contributions:**
1. Implemented state-of-the-art CNN for agricultural disease detection
2. Achieved near-perfect accuracy suitable for real-world deployment
3. Demonstrated transfer learning effectiveness for limited agricultural datasets
4. Provided comprehensive evaluation and error analysis

**Limitations:**
- Dataset from controlled conditions (uniform backgrounds)
- Limited to 9 plant species
- Requires further validation on field-captured images
- Single-disease assumption (no multi-disease detection)

**Future Work:**
1. Expand to more plant species relevant to Nigerian agriculture
2. Test on field-captured images with varied backgrounds
3. Develop mobile application for farmer access
4. Integrate treatment recommendations
5. Add multi-disease detection capability

**Impact:**
This system demonstrates AI's potential to address agricultural challenges in Nigeria and sub-Saharan Africa, providing farmers with accessible, accurate disease diagnosis to improve crop yields and food security.

---

## Contact Information

**Project Supervisor:**  
Ayuba Muhammad  
Email: muhammadayubaxy@gmail.com  
Affiliation: New Horizons ICT

**Team Lead:**  
Osemudiame Okoeguale  
Email: iamose@aol.com

**Institution:**  
Department of Information & Communication Technology  
Bells University of Technology  
Ota, Ogun State, Nigeria

---




