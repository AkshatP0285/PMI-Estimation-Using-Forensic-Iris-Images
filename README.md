<!-- Typing animation -->
<div align="center">
  <img src="https://readme-typing-svg.demolab.com?font=Fira+Code&weight=500&size=24&pause=1000&color=F7F7F7&center=true&vCenter=true&width=700&height=45&lines=Hi+there+%F0%9F%91%8B+I'm+Akshat+Pandey;PMI+Estimation+using+Forensic+Iris+Images" alt="Typing SVG" />
</div>

---

# 🧠 Post-Mortem Interval Estimation using Forensic Iris Images

This project explores **deep learning-based Post-Mortem Interval (PMI) estimation** using **RGB and NIR iris images** of deceased individuals. Built as part of my **Bachelor’s Thesis at IISER Bhopal**, this work benchmarks both classical CNNs and modern vision transformer architectures across realistic forensic scenarios.

---

## 🔍 Problem Statement

**PMI** refers to the time elapsed since death. Accurate estimation is crucial for:

- Time of death (TOD) analysis in forensics  
- Biometric identification of missing persons  
- Automation of forensic workflows using iris data

While physical and biochemical cues are traditionally used for PMI, biometric iris images remain underutilized. This work aims to fill that gap using **deep learning models trained on post-mortem iris images**.

---

## 🧪 Research Contributions

- 📊 **First-of-its-kind benchmarking** of Vision Transformers (DINO, CLIP) and CNNs (InceptionV3, DenseNet121) on PMI regression  
- 🔍 Explored **RGB**, **NIR**, and **multispectral (RGB+NIR)** inputs  
- ⚙️ Built pipelines for **subject-disjoint**, **sample-disjoint**, and **cross-dataset** evaluations  
- 🧠 Proposed MLP-based regression using transformer embeddings for improved generalization

---

## 🗃️ Dataset Overview

| Dataset         | Type      | Subjects | Images | Spectrum  | PMI Range (hours) |
|----------------|-----------|----------|--------|-----------|-------------------|
| Warsaw Biobase | Real      | 79       | 4,866  | RGB + NIR | 5 – 814           |
| NIJ Dataset | Real | 1000/class × 18 | 10,413 | RGB + NIR  | 0 – 1674          |
| Synthetic Set  | Simulated | 1000/class × 18 | 180,000 | NIR only  | 0 – 1674          |


🔍 **Challenge**: Dataset is highly imbalanced — majority samples lie in lower PMI ranges. Iris degradation at high PMIs causes severe loss of texture, impacting prediction.

---

## 🧠 Methodology

### 📸 CNN-Based Models
- Backbones: `InceptionV3`, `DenseNet121`
- Separate models for RGB, NIR, and multispectral images
- Used fully connected regression head to predict continuous PMI

### 🧠 Vision Transformer Models
- Feature extractors: `CLIP ViT-B/16`, `DINOv2`
- Frozen vision encoder → feature vector → MLP regressor
- Trained on RGB, NIR, and combined embeddings

---

## 🧪 Evaluation Strategies

| Split Type        | Description                                                  |
|-------------------|--------------------------------------------------------------|
| Sample-wise       | Same subjects, different samples in train/test               |
| Subject-wise      | Different subjects from same dataset                         |
| Cross-dataset     | Training and testing on different datasets (most realistic)  |

---

## 📊 Results Summary

| Model        | Spectrum     | Split Type     | RMSE (h) | MAE   | Notes                                |
|--------------|--------------|----------------|----------|-------|--------------------------------------|
| InceptionV3  | RGB          | Subject-wise   | ~22.7    | 18.5  | Outperformed DenseNet in most cases |
| DINOv2 + MLP | NIR          | Cross-dataset  | Lower RMSE at high PMIs |  | More robust to image degradation     |
| CLIP         | RGB          | Sample-wise    | Higher MAE |       | Less stable than DINO                |

> 📌 **DINO** showed better generalization on high PMI ranges. **InceptionV3** had better performance in early PMI ranges. Multispectral models worked best in subject-disjoint settings.

---

## ⚠️ Challenges

- 📉 High RMSE for high-PMI images due to iris degradation  
- 📊 Dataset imbalance: majority samples in early PMI classes  
- 🌫️ Environmental degradation blurs and occludes iris structures

---

## 🚀 Future Work

- 🧬 Combine CNN and transformer features for robust PMI estimation across all ranges  
- 🎨 Generate synthetic **RGB iris images** to balance high PMI ranges  
- ⚡ Apply **weighted fusion** strategies for multispectral inputs

---

## 🔧 Getting Started

```bash
git clone https://github.com/AkshatP0285/PMI-Estimation-From-Iris
cd PMI-Estimation-From-Iris
pip install -r requirements.txt
