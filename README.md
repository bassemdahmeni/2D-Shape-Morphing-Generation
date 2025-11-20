# Morphing2D: Conditional VAE for Shape Morphing Generation

[![Python](https://img.shields.io/badge/Python-3.8%252B-blue)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%252B-orange)](https://pytorch.org/)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.5%252B-green)](https://opencv.org/)

A deep learning framework that generates smooth 2D shape morphing sequences using **Conditional Variational Autoencoders (CVAE)**, trained on SRVF-generated ground truth data.

---

## 🎯 Project Overview
Morphing2D provides a complete pipeline for **learning-based 2D shape morphing**:

1. **Data Generation** – Extract contours and generate morphing sequences using **SRVF** (Square-Root Velocity Function).
2. **Model Training** – Train a **Conditional VAE** to generate morphing sequences from source-target pairs.
3. **Evaluation** – Compare the learned morphing sequences against the SRVF mathematical baseline.

---

## 📁 Project Files

### 1. `morphing_2d_Data_generation.ipynb`
**Purpose:** Generate the training dataset with SRVF ground truth.

**Process:**
- Loads 2D geometric shapes from input data.
- Extracts contours using **Canny edge detection**.
- Generates optimal morphing sequences with SRVF.
- Saves a structured JSON dataset containing:
  - `source_contour` – starting shape (200 points, 2D)
  - `target_contour` – ending shape (200 points, 2D)
  - `morphing_sequence` – SRVF-generated ground truth (7 frames)

**Output:** `morphing_dataset.json` with 28,270 training samples.

---

### 2. `CVAE_for_morphing_generation.ipynb`
**Purpose:** Train and evaluate the Conditional VAE model.

**Process:**
- Loads the JSON dataset from the previous step.
- Implements **Conditional VAE architecture**:
  - **Encoder:** Maps sequences to latent space.
  - **Decoder:** Generates sequences from latent codes + conditions.
  - **Condition:** Concatenated source + target contours.
- Trains using **MSE loss** against SRVF sequences.
- Evaluates morphing using **optical flow metrics**.
- Compares generated sequences with SRVF baseline.

**Output:** Trained CVAE model and smoothness analysis.

---

## 🚀 Quick Start

1. **Generate Training Data**
```python
# Run the notebook to generate dataset
morphing_2d_Data_generation.ipynb
```
Input: /kaggle/input/2d-geometric-shapes-17-shapes/
Output: morphing_dataset.json containing SRVF morphing sequences.
2. **Train CVAE Model**
```python
# Run the notebook to train the CVAE
CVAE_for_morphing_generation.ipynb
```
Loads the dataset, trains the model, generates new morphing sequences, and evaluates smoothness.

## 📊 Performance Results

Smoothness Comparison (Lower is Better)

Method	Spatial Smoothness	Temporal Smoothness
SRVF (GT)	1.60	0.000001
CVAE (Ours)	2.91	0.000064

Metric Definitions:

Spatial Smoothness: Coherence of movement in space.

Temporal Smoothness: Speed consistency between frames.

## 🔬 Technical Approach

Supervised Learning Framework

Input: Source + target contours (800D)

Ground Truth: SRVF-generated sequences (7 frames × 200 points × 2D)

Model: Conditional VAE with 64D latent space

Loss: MSE reconstruction + KL divergence

Data Generation

Contour extraction using OpenCV (resampled to 200 points)

SRVF interpolation for mathematically optimal smoothness

Dataset: 28,270 source-target morphing pairs


## 🛠️ Dependencies
```
# Core libraries
python >= 3.8
torch >= 2.0.0
numpy >= 1.21.0
opencv-python >= 4.5.0
matplotlib >= 3.5.0
scipy >= 1.7.0

# Additional utilities
scikit-image
tqdm
```
## 📈 Key Insights

SRVF Advantage: Guarantees mathematically optimal smoothness.

CVAE Strength: Learns morphing generation without manual correspondence.

Trade-off: CVAE provides flexibility, SRVF ensures perfect smoothness.
