# GravGlitch Classifier

## Project Overview

**Project Title:** GlitchNet: CNN for GW Detector Glitch Classification

**Description:**  
This project implements a convolutional neural network (CNN) with residual connections to classify transient noise artifacts ("glitches") in data from gravitational-wave (GW) detectors (e.g., LIGO/Virgo) using images of the spectrogram plots. Glitches can mimic or obscure true astrophysical signals, and accurate classification is critical for improving detector sensitivity and analysis pipelines.

## Model Architecture

1. **Initial Feature Extraction**  
   - 7×7 convolution (stride=2) + BatchNorm + ReLU  
   - 3×3 max-pooling (stride=2)  

2. **Residual Blocks**  
   - Two stacked `ResidualBlock` units with 3×3 convolutions (padding=1).  
   - Identity skip connections for gradient stabilization.  

3. **Global Aggregation & Classification**  
   - Adaptive average pooling → 1×1 feature vector.  
   - Fully connected layer for `num_classes` logits.  

**Framework**: PyTorch 

**Why This Architecture?**
- **Stable Training:** Residual connections ensure robust gradient flow, speeding up convergence and improving accuracy on limited glitch datasets.
- **Model Depth vs. Complexity:** A two-block residual design balances depth (for feature richness) with computational efficiency, making training feasible on standard GPUs. Additionally, the relatively limited size of the dataset discourages using a very deep network, as it could lead to overfitting. This architecture is therefore well-suited for extracting meaningful patterns while avoiding excessive model complexity.

- *Framework:* PyTorch

## Dataset

- We use the Gravity Spy dataset which is a citizen scientist program for glitch classification in GW Detectors. The dataset was downloaded from Kaggle: [Gravity Spy (Gravitational waves)](https://www.kaggle.com/datasets/tentotheminus9/gravity-spy-gravitational-waves)