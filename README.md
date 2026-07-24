---
title: CosmicVision
colorFrom: blue
colorTo: gray
sdk: gradio
app_file: app.py
pinned: false
---

# Sequence-Based Classification of Astronomical Objects Using Deep Learning

**Project Name:** CosmicVision  
**Program/Event:** Summer Siege × Odyssey 2026  
**Institution:** IIT Gandhinagar  
 
---

CosmicVision is an interactive astronomical image classification web application built using EfficientNet-B3 and Grad-CAM explainability.

The application allows users to upload an astronomical image and receive a predicted celestial object category along with confidence scores and a Grad-CAM visualization.

## Supported Classes

The model classifies astronomical images into five categories:

- Elliptical Galaxy
- Nebula
- Planetary Object
- Spiral Galaxy
- Star Cluster

## Features

- Upload an astronomical image
- Predict the primary celestial object class
- Display confidence scores for all five classes
- Generate Grad-CAM visualization to highlight important image regions
- Show inference time for the uploaded image

## Model Details

The classification model is based on EfficientNet-B3 with ImageNet pretrained weights. It was trained using a two-stage transfer learning pipeline:

1. Stage 1: The pretrained EfficientNet-B3 backbone was frozen and only the custom classifier head was trained.
2. Stage 2: The full model was fine-tuned using differential learning rates.

## Final Performance

The final model achieved the following results on the recreated held-out test set:

| Metric | Value |
|---|---:|
| Test Accuracy | 93.51% |
| Macro F1-score | 90.62% |
| Weighted F1-score | 93.52% |

## Explainability

Grad-CAM is used to visualize the regions of the uploaded astronomical image that contributed most strongly to the model's prediction. This helps make the model's decision more interpretable by showing whether it focuses on meaningful object regions such as galaxy cores, spiral arms, nebular structures, compact clusters, or planetary disks.

## Files

This Space uses the following main files:

- `app.py`: Gradio application code
- `best_model_stage2.pth`: Final trained EfficientNet-B3 model checkpoint
- `class_mapping.json`: Class label mapping
- `requirements.txt`: Python dependencies

---

**IIT Gandhinagar**
**Vipra Madanala**
**B.Tech AI 2029**