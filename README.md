
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
