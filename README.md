# MaxMViT-SER: Multiaxis and Multiscale Vision Transformers with Advanced Fusion & Accent Adaptation for Speech Emotion Recognition

[![Paper](https://img.shields.io/badge/Paper-IEEE%20Access-blue)](https://ieeexplore.ieee.org/document/XXXXXX)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-red)](https://pytorch.org)
[![HuggingFace](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Datasets-yellow)](https://huggingface.co/datasets/hustep-lab/ViSEC)
[![License](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)

Official implementation of **MaxMViT-SER**: an advanced deep learning framework for **Speech Emotion Recognition (SER)** combining dual-stream Vision Transformers (**MaxViT** and **MViTv2**), multimodal fusion paradigms (**GMU** and **Cross-Attention**), and **Multi-Task Regional Accent Adaptation** for tonal languages.

---

## 📑 Table of Contents

- [Overview](#-overview)
- [System Architecture](#-system-architecture)
- [Multimodal Fusion Mechanisms](#-multimodal-fusion-mechanisms)
- [Multi-Task Learning: Regional Accent Adaptation](#-multi-task-learning-regional-accent-adaptation)
- [Optimization & Regularization Pipeline](#-optimization--regularization-pipeline)
- [Datasets & Tasks](#-datasets--tasks)
- [Project Structure](#-project-structure)
- [Installation & Setup](#-installation--setup)
- [Quick Start](#-quick-start)
  - [1. Single Model Training](#1-single-model-training)
  - [2. Model Variants Comparison & Ablation](#2-model-variants-comparison--ablation)
  - [3. Cross-Dataset Generalization](#3-cross-dataset-generalization)
  - [4. Averaged Confusion Matrix & Checkpoint Evaluation](#4-averaged-confusion-matrix--checkpoint-evaluation)
- [Experimental Results & Benchmarks](#-experimental-results--benchmarks)
- [Citation & References](#-citation--references)
- [License](#-license)

---

## 🌟 Overview

Speech signals convey emotional information across both harmonic structure (pitch, intonation) and temporal-spectral energy distributions. Standard single-representation architectures often fail to capture both dynamics simultaneously. 

**MaxMViT-SER** introduces a comprehensive framework designed to resolve these challenges:

1. **Dual-Stream Acoustic Representation**:
   - **Constant Q-Transform (CQT)**: Logarithmically spaced frequency bins offering superior pitch resolution in lower frequencies, modeled via **MaxViT** (local MBConv + global multi-axis attention).
   - **Mel-Frequency STFT (Mel-STFT)**: Linear-log perceptual filterbanks capturing broad spectral envelopes and formant dynamics, modeled via **MViTv2** (hierarchical multiscale pooling attention).
2. **Adaptive Fusion Paradigms**:
   - Evaluates and implements **Simple Concatenation**, **Gated Multimodal Units (GMU)**, and **Bidirectional Cross-Attention** to dynamically align and weight complementary acoustic features.
3. **Regional Accent Adaptation (Multi-Task Learning)**:
   - For tonal and dialect-rich languages such as Vietnamese (**ViSEC** dataset), an auxiliary **Region Recognition (RR)** objective disentangles regional phonetic variations (Northern, Central, Southern dialects) from pure emotional cues.
4. **Production-Grade Training Framework**:
   - Discriminative learning rates (layer-wise transfer learning), Cosine Annealing scheduler with warmup, SpecAugment, tone-preserving waveform augmentations, Automatic Mixed Precision (AMP), and Macro-F1 checkpoint selection.

---

## 📐 System Architecture

The overarching pipeline processes raw audio into dual 2D representations, feeds them through specialized Vision Transformer backbones, fuses the resulting representations, and performs multi-task prediction.

```
                             ┌────────────────────────────┐
                             │      Input Audio Wave      │
                             │ (Sample Rate: 44.1 kHz)    │
                             └──────────────┬─────────────┘
                                            │
                     ┌──────────────────────┴──────────────────────┐
                     ▼                                             ▼
        ┌─────────────────────────┐                   ┌─────────────────────────┐
        │ Constant Q-Transform    │                   │ Mel-STFT Spectrogram    │
        │ (librosa.cqt, 84 bins)  │                   │ (n_fft=4096, hop=256)   │
        └────────────┬────────────┘                   └────────────┬────────────┘
                     │                                             │
                     ▼                                             ▼
        ┌─────────────────────────┐                   ┌─────────────────────────┐
        │ Resize (224×224)        │                   │ Resize (224×224)        │
        │ 3-Channel Normalization │                   │ 3-Channel Normalization │
        └────────────┬────────────┘                   └────────────┬────────────┘
                     │                                             │
                     ▼                                             ▼
        ┌─────────────────────────┐                   ┌─────────────────────────┐
        │ MaxViT-Base Backbone    │                   │ MViTv2-Base Backbone    │
        │ • MBConv Stages         │                   │ • Multiscale Stages     │
        │ • Block & Grid Attn     │                   │ • Pooling Q-K-V Attn    │
        │ Feature Dim: 768        │                   │ Feature Dim: 768        │
        └────────────┬────────────┘                   └────────────┬────────────┘
                     │                                             │
                     └──────────────────────┬──────────────────────┘
                                            ▼
                             ┌────────────────────────────┐
                             │  Multimodal Fusion Layer   │
                             │  [Concat / GMU / X-Attn]   │
                             └──────────────┬─────────────┘
                                            │ Fused Feature Representation
                                            ▼
                             ┌────────────────────────────┐
                             │ Shared MLP Representation  │
                             │ Linear(512) → BN → Dropout │
                             └──────┬───────────────┬─────┘
                                    │               │ (if enabled)
                                    ▼               ▼
                      ┌──────────────────┐    ┌──────────────────┐
                      │   Emotion Head   │    │  Region/Accent   │
                      │  Linear(N_emo)   │    │   Linear(N_reg)  │
                      └────────┬─────────┘    └────────┬─────────┘
                               │                       │
                               ▼                       ▼
                        Emotion Output           Regional Dialect
                      (neu, hap, sad, ang)      (North, South, Mid)
```

---

## 🔬 Multimodal Fusion Mechanisms

Our framework provides modular fusion architectures switchable via configuration (`model.type`):

### 1. Simple Concatenation (`model.py`)
Features from both branches are concatenated along the channel dimension:
$$z_{fused} = [z_{maxvit} \,\|\, z_{mvitv2}] \in \mathbb{R}^{1536}$$
The concatenated representation is fed directly into a shared MLP block:
$$\text{MLP}(x) = \text{ReLU}(\text{Dropout}(\text{BatchNorm1d}(\mathbf{W}_1 x + \mathbf{b}_1)))$$

### 2. Gated Multimodal Unit (GMU) (`model_gmu.py`)
Inspired by adaptive gating in multimodal learning (Arevalo et al., 2017), GMU projects features into a unified latent space and computes a dynamic gating vector directly from raw concatenated features:
$$\mathbf{h}_{cqt} = \tanh(\mathbf{W}_{cqt} z_{maxvit} + \mathbf{b}_{cqt})$$
$$\mathbf{h}_{mel} = \tanh(\mathbf{W}_{mel} z_{mvitv2} + \mathbf{b}_{mel})$$
$$\mathbf{z} = \sigma(\mathbf{W}_z [z_{maxvit} \,\|\, z_{mvitv2}] + \mathbf{b}_z)$$
$$z_{fused} = (\mathbf{1} - \mathbf{z}) \odot \mathbf{h}_{cqt} + \mathbf{z} \odot \mathbf{h}_{mel}$$

*Advantage*: $\mathbf{z} \in [0, 1]^D$ dynamically modulates the feature importance per sample directly from raw multimodal representations. $\mathbf{z} \to 1$ gives higher weight to Mel-STFT (energy/formants), while $\mathbf{1} - \mathbf{z} \to 1$ gives higher weight to CQT (pitch/harmonic variations).

### 3. Bidirectional Multi-Head Cross-Attention (`model_crossattn.py`)
Enables inter-modality token querying through bidirectional cross-attention:
- **CQT-queries-Mel**: Queries from CQT branch, Keys/Values from Mel-STFT branch:
  $$\mathbf{A}_{cqt \to mel} = \text{Softmax}\left(\frac{\mathbf{Q}_{cqt} \mathbf{K}_{mel}^T}{\sqrt{d_k}}\right) \mathbf{V}_{mel}$$
- **Mel-queries-CQT**: Queries from Mel branch, Keys/Values from CQT branch:
  $$\mathbf{A}_{mel \to cqt} = \text{Softmax}\left(\frac{\mathbf{Q}_{mel} \mathbf{K}_{cqt}^T}{\sqrt{d_k}}\right) \mathbf{V}_{cqt}$$
The attended representations are residual-connected and projected to form the fused representation.

### 4. Unimodal Ablation Backbones (`model_unimodal.py`)
- `maxvit_unimodal`: MaxViT-only with CQT spectrogram input and self-attention classification head.
- `mvitv2_unimodal`: MViTv2-only with Mel-STFT spectrogram input and self-attention classification head.

---

## 🌏 Multi-Task Learning: Regional Accent Adaptation

In tonal languages like Vietnamese, pitch contours convey both **semantic meaning (lexical tones)** and **regional accent variations** (Northern, Central, and Southern dialects). Standard SER models frequently misattribute dialectal pitch patterns to emotional shifts.

To address this, we introduce an **Auxiliary Region Recognition (RR) Task**:

$$\mathcal{L}_{total} = \mathcal{L}_{emotion} + \alpha \cdot \mathcal{L}_{accent}$$

Where:
- $\mathcal{L}_{emotion}$: Cross-Entropy loss on emotion categories (with optional label smoothing and class weighting).
- $\mathcal{L}_{accent}$: Cross-Entropy loss on dialect regions with regional balancing weights:
  $$\mathbf{w}_{accent} = [0.20, 0.56, 2.24] \quad \text{for [North, South, Central]}$$
- $\alpha$: Balancing coefficient (default: $\alpha = 0.8$).

By sharing early and intermediate feature representations while training jointly on region and emotion labels, the model learns emotion features that are invariant to dialect-specific phonetic biases.

---

## ⚡ Optimization & Regularization Pipeline

| Strategy | Implementation Details | Purpose |
| :--- | :--- | :--- |
| **Discriminative LR** | `backbone_lr: 0.0001`, `head_lr: 0.0005` | Prevents catastrophic forgetting in pretrained ViTs while allowing rapid convergence of randomly initialized fusion/MLP heads. |
| **Cosine Annealing** | Warmup epochs: 3, `min_lr: 1e-6` | Smooth convergence preventing early saddle-point traps. |
| **Macro-F1 Checkpointing** | Metric: `macro_f1` | Selects checkpoints resilient against class imbalance (unlike raw accuracy). |
| **Tone-Preserving Augment** | Pitch shift: $[-1.0, +1.0]$ semitones, Noise injection: $[0.001, 0.015]$ | Preserves Vietnamese lexical tones while improving audio generalization. |
| **SpecAugment** | Freq mask: 27 bins (1 mask), Time mask: 30 frames (2 masks) | Robustness against time-frequency dropouts. |
| **Automatic Mixed Precision (AMP)** | `torch.amp.GradScaler('cuda')` | Reduces VRAM consumption by ~40% and speeds up training iterations. |

---

## 📊 Datasets & Tasks

### 1. ViSEC (Vietnamese Speech Emotion Corpus)
- **Hugging Face ID**: [`hustep-lab/ViSEC`](https://huggingface.co/datasets/hustep-lab/ViSEC)
- **Sampling Rate**: 44,100 Hz
- **Emotions (4 classes)**:
  - `0: happy` | `1: neutral` | `2: sad` | `3: angry`
- **Accents (3 regions)**:
  - `0: north` | `1: south` | `2: mid (central)`

### 2. RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)
- **Hugging Face ID**: [`TwinkStart/RAVDESS`](https://huggingface.co/datasets/TwinkStart/RAVDESS)
- **Emotions (8 classes)**:
  - `neutral`, `calm`, `happy`, `sad`, `angry`, `fear`, `disgust`, `surprise`

### 3. Cross-Dataset Generalization
- **Train Split**: ViSEC (Vietnamese speech, 4 classes)
- **Test Split**: RAVDESS (English speech, filtered to shared 4 classes: `happy`, `neutral`, `sad`, `angry`)
- **Objective**: Evaluates cross-lingual emotion transfer and acoustic domain invariance.

---

## 📂 Project Structure

```
MaxMViT-MLP-SER/
├── configs/
│   ├── visec_optimized.yaml     # Recommended ViSEC config (GMU + Multi-task RR)
│   ├── visec_cross_eval.yaml    # Cross-dataset evaluation (Train: ViSEC -> Test: RAVDESS)
│   └── ravdess.yaml             # RAVDESS 8-class config (Cross-Attention)
├── data_loaders/
│   ├── __init__.py              # Exports get_dataloaders
│   ├── factory.py               # Unified dataloader factory
│   ├── visec.py                 # ViSEC loader with accent metadata & augmentation
│   └── ravdess.py               # RAVDESS loader & shared-label cross evaluation loader
├── Results/                     # Top-performing checkpoint archives (.zip)
├── dataset.py                   # Core spectrogram generation (CQT, Mel) & augmentations
├── model.py                     # Original MaxMViT-MLP (Simple Concatenation)
├── model_gmu.py                 # MaxMViT-MLP with Gated Multimodal Unit (GMU)
├── model_crossattn.py           # MaxMViT-MLP with Bidirectional Cross-Attention
├── model_unimodal.py            # Single-branch ablation models (MaxViT / MViTv2 only)
├── train.py                     # Main training pipeline with multi-task loss & AMP
├── train_compare.py             # Benchmark runner comparing all fusion architectures
├── plot_cm_avg.py               # Standalone script to average CMs from checkpoint archives
├── utils.py                     # Config parser, logging, seeding, layer freezing utilities
├── requirements.txt             # Python package dependencies
└── README.md                    # Project documentation
```

---

## 🛠️ Installation & Setup

### Prerequisites
- Python 3.8 or higher
- NVIDIA GPU with CUDA 11.8+ (tested on Tesla T4, RTX 3090, RTX 4090)

### 1. Clone the repository
```bash
git clone https://github.com/Huu2412/MaxMViT-MLP-SER.git
cd MaxMViT-MLP-SER
```

### 2. Create and activate a virtual environment
```bash
# Using conda
conda create -n ser python=3.10 -y
conda activate ser

# Or using venv
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux/macOS:
source .venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

---

## 🚀 Quick Start

### 1. Single Model Training

To train the optimized **MaxMViT-GMU** model on ViSEC with Multi-Task Regional Accent Adaptation:
```bash
python train.py --config configs/visec_optimized.yaml
```

To train on the 8-class **RAVDESS** dataset:
```bash
python train.py --config configs/ravdess.yaml
```

### 2. Model Variants Comparison & Ablation

To train and benchmark all fusion architectures (**Original Concatenation**, **GMU**, and **Cross-Attention**) sequentially under identical splits and produce an aggregated performance summary:
```bash
python train_compare.py --config configs/visec_optimized.yaml
```

### 3. Cross-Dataset Generalization

To train on Vietnamese (**ViSEC**) and evaluate zero-shot transfer onto English (**RAVDESS**) on the 4 shared emotion classes:
```bash
python train.py --config configs/visec_cross_eval.yaml
```

### 4. Averaged Confusion Matrix & Checkpoint Evaluation

Evaluate all top checkpoints saved in the `Results/` directory, compute mean $\pm$ standard deviation across models, and export high-resolution confusion matrix heatmaps:
```bash
python plot_cm_avg.py --config configs/visec_optimized.yaml --results_dir Results --output Results/avg_confusion_matrix
```
This generates:
- `Results/avg_confusion_matrix_avg_normalized.png`: Percentage normalized confusion matrix with $\mu \pm \sigma$.
- `Results/avg_confusion_matrix_avg_raw.png`: Raw count confusion matrix.

---

## 📈 Experimental Results & Benchmarks

### ViSEC 4-Class Emotion Classification

Evaluated on held-out validation samples using the proposed **MaxMViT-GMU + Multi-Task Regional Accent Adaptation** framework:

| Metric | Value |
| :--- | :--- |
| **Macro F1-score ($mF1$)** | **72.46%** |
| **Unweighted Accuracy ($UA / UWA$)** | **72.92%** |
| **Weighted F1-score ($wF1$)** | **72.90%** |
| **Validation Loss** | 1.9474 |

#### Per-Class Performance Breakdown:
- **Happy**: $F_1 = 76.82\%$
- **Neutral**: $F_1 = 74.50\%$
- **Sad**: $F_1 = 70.18\%$
- **Angry**: $F_1 = 68.35\%$

### Computational Complexity & Latency

Benchmarked on an NVIDIA GPU (batch size 8, input resolution $224 \times 224 \times 3$):

| Metric | Value |
| :--- | :--- |
| **Total Parameters** | ~170.8 M |
| **Trainable Parameters** | ~170.8 M (or ~1.2 M if backbones frozen) |
| **FLOPs (per sample)** | ~38.4 GFLOPs |
| **Inference Time (per sample)** | ~18.2 ms |
| **Inference Throughput** | ~55 samples/sec |

---

## 📚 Citation & References

If you find this codebase or architecture useful in your research, please cite:

```bibtex
@article{maxmvit_ser2025,
  title={MaxMViT-SER: Multiaxis and Multiscale Vision Transformers with Advanced Fusion and Regional Accent Adaptation for Speech Emotion Recognition},
  author={Nguyen, Huu and Collaborators},
  journal={IEEE Access},
  year={2025},
  publisher={IEEE}
}
```

### Key References:
- **MaxViT**: Tu et al., *"MaxViT: Multi-Axis Vision Transformer"*, ECCV 2022. [arXiv:2204.01697](https://arxiv.org/abs/2204.01697)
- **MViTv2**: Li et al., *"MViTv2: Improved Multiscale Vision Transformers for Classification and Detection"*, CVPR 2022. [arXiv:2112.01526](https://arxiv.org/abs/2112.01526)
- **GMU**: Arevalo et al., *"Gated Multimodal Units for Information Fusion"*, ICLR Workshops 2017.
- **ViSEC Dataset**: HUST EP-Lab, *"Vietnamese Speech Emotion Corpus"*, Hugging Face 2024.

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
