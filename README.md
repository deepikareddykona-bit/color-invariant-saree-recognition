
# Color-Invariant Saree Design Recognition

## Overview

This project implements a color-invariant saree design recognition system using deep metric learning and PyTorch.

The goal is to identify saree designs based on their surface patterns while remaining robust to changes in color palette.

## Problem

Given a query saree image, the system compares it with a gallery of known designs and:

- Identifies the most similar design.
- Ranks gallery designs using similarity scores.
- Verifies whether two saree images contain the same design.
- Reduces dependence on color so that the same motif in different palettes can still match.

## Approach

The system uses an ImageNet-pretrained ConvNeXt-Atto backbone followed by GeM pooling and a 256-dimensional L2-normalized embedding.

Training uses multiple augmented views of each image, including synthetic recoloring, random crops, flips, rotations, blur, lighting changes, noise and JPEG artifacts. Supervised contrastive learning encourages images of the same design to have similar embeddings while separating different designs.

During inference, cosine similarity is used to rank gallery images and perform verification.

## Model Pipeline

```text
Input RGB Image
       ↓
Image Preprocessing
       ↓
Color / Geometric Augmentation
       ↓
ConvNeXt-Atto
       ↓
GeM Pooling
       ↓
256-D Embedding
       ↓
L2 Normalization
       ↓
Cosine Similarity
       ↓
Identification / Verification
## Evaluation Results

The model was evaluated for both identification and verification.

| Metric | Result |
|---|---:|
| P1 R@1 | 0.889 |
| P1 mAP | 0.936 |
| P1 R@1 (Held-out Family) | 0.842 |
| P2 R@1 | 0.874 |
| P2 Trap@1 | 0.000 |
| P3 R@1 | 0.923 |
| P4 R@1 | 0.556 |
| Verification AUC | 0.987 |
| Verification EER | 0.054 |
| TAR @ FAR = 1% | 0.679 |
| Same-Palette AUC | 0.987 |
| Verification Accuracy | 0.949 |

### Baseline Comparison

The proposed color-invariant embedding model was compared against zero-shot RGB and grayscale baselines.

| Model | P1 R@1 | P1 mAP | P3 R@1 | Verification AUC |
|---|---:|---:|---:|---:|
| Proposed ConvNeXt-Atto | 0.889 | 0.936 | 0.923 | 0.987 |
| Zero-shot Grayscale | 0.599 | 0.699 | 0.702 | 0.874 |
| Zero-shot RGB | 0.459 | 0.561 | 0.551 | 0.810 |

## Implementation Note

The implementation was developed with reference to the public color-invariant saree recognition implementation and adapted/run for this evaluation. The proprietary DeepLure dataset is not included in this repository.

## Dataset

The DeepLure dataset is proprietary and is therefore not redistributed in this repository.

The notebook expects the dataset to be available in the Kaggle environment.

## Limitations

- Performance on real re-photographed sarees is lower than on controlled evaluation settings.
- The dataset does not provide explicit design labels for every image.
- Verification performance depends on the selected validation threshold.

## Future Work

- Improve robustness to real-world lighting and camera variations.
- Explore stronger transformer-based architectures.
- Add larger-scale hard-negative mining.
- Improve real re-photo performance.
- Deploy the embedding model as an efficient similarity-search service.
