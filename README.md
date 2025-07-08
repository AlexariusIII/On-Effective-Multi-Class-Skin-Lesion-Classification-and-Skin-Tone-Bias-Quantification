# On Effective Multi-Class Skin Lesion Classification and Skin-Tone Bias Quantification

**Official code for the paper:**
**"On Effective Multi-Class Skin Lesion Classification and Skin-Tone Bias Quantification"**

---

## Overview

This repository provides the code and processing scripts for robust multi-class skin lesion classification and the quantification of skin tone bias in dermatological AI. We harmonize 13 public datasets, develop a deep learning-based skin tone classifier, and analyze model performance and fairness across skin tone groups.

**Key contributions:**
- **Comprehensive Data Harmonization:** Unified 13 public datasets into six clinically relevant skin lesion categories.
- **Skin Tone Classification:** Developed a deep learning model for skin tone estimation, enabling pseudo-labeling of unannotated images.
- **Bias Quantification:** Quantified and analyzed model performance across skin tone groups, revealing counterintuitive trends and highlighting the need for balanced datasets.

---

## Table of Contents
- [Background](#background)
- [Dataset Overview](#dataset-overview)
- [Workflow](#workflow)
- [Experiments & Results](#experiments--results)
- [Installation](#installation)
- [Usage](#usage)
- [License & Funding](#license--funding)
- [Contact](#contact)

---

## Background

Accurate multi-class skin lesion classification is critical for effective patient care, but is often hindered by limited harmonized data and unaddressed skin tone biases. Our work presents a large-scale analysis using a harmonized dataset from 13 public sources, and introduces a deep learning approach for both skin tone and lesion classification, with a focus on fairness and bias quantification.

---

## Dataset Overview

We harmonized 13 publicly available datasets, mapping their diverse labels into six standardized categories:
- **Basal Cell Carcinoma (BCC)**
- **Squamous Cell Carcinoma (SCC)**
- **Actinic Keratosis (ACK)**
- **Seborrheic Keratosis (SEK)**
- **Melanoma (MEL)**
- **Nevus (NEV)**

Skin tone information (where available) was mapped to three categories:
- **A:** Fitzpatrick I/II (light)
- **B:** Fitzpatrick III/IV (medium)
- **C:** Fitzpatrick V/VI (dark)

See the paper for a full dataset table and label mapping.

---

## Workflow

1. **Data Processing:** Harmonize and standardize all datasets for both skin tone and multi-class classification.
2. **Skin Tone Classifier Training:** Train a deep learning model on Fitzpatrick-annotated images.
3. **Pseudo-Label Generation:** Use the trained classifier to pseudo-label skin tone for unannotated images.
4. **Multi-Class Classifier Training:** Train models for six-way skin lesion classification.
5. **Bias Quantification:** Evaluate and analyze model performance across skin tone groups.

---

## Experiments & Results

- **Architectures:** Vision Transformers (ViT-Small, ViT-Base), ConvNeXt (Small, Base), all pretrained on ImageNet.
- **Training:** AdamW optimizer, ReduceLROnPlateau scheduler, 50 epochs, early stopping on F1-score, 5-fold cross-validation.
- **Augmentation:** Random resized crop, flips, rotation, color jitter, normalization, random erasing.
- **Metrics:** Accuracy, F1 Score, AUROC, Average Precision.

**Key findings:**
- Strong multi-class classification performance (F1 up to 0.84, AUROC up to 0.98).
- **Counterintuitive bias:** Models performed better on darker skin tones (B, C) than on lighter tones (A), despite severe underrepresentation of darker tones in the data.
- **Call to action:** Results highlight the urgent need for more balanced, human-validated datasets to ensure fair and reliable AI for all skin types.

See the paper for detailed tables, figures, and discussion.

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/AlexariusIII/On-Effective-Multi-Class-Skin-Lesion-Classification-and-Skin-Tone-Bias-Quantification.git
   cd On-Effective-Multi-Class-Skin-Lesion-Classification-and-Skin-Tone-Bias-Quantification
   ```

2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   (Or use your preferred environment manager.)

3. **Download datasets:**  
   All datasets used are publicly available. See the paper and `data/` directory for download scripts and instructions.

---

## Usage

- **Training a skin tone classifier:**
  ```bash
  python train.py --task skin_tone  --arch convnext_small
  ```

- **Training a multi-class classifier:**
  ```bash
  python train.py --task multiclass  --arch vit_small
  ```

- **Skin tone inference and pseudo-labeling:**
  ```bash
  python skin_tone_inference.py
  ```

- **Evaluation and bias quantification:**  
  See `classification_evaluation.py` and the paper for details.

---

## Funding

- **Funding:** This work was supported by ‘KITE’ (Plattform für KI-Translation Essen), the REACT-EU initiative (EFRE-0801977), the KIADEKU Project (V6KIP039), and the Cancer Research Center Cologne Essen (CCCE).

---

**Data Availability:**  
All data used are publicly accessible. See the paper and repo for download and harmonization instructions.

---

**Acknowledgements:**  
We thank all dataset providers and contributors.