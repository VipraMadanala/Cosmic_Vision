This folder contains the main notebooks used for the CosmicVision project.

## 1. CosmicVision.ipynb

This notebook contains the main training pipeline for the project. It includes:

- Dataset collection and organization
- Image preprocessing and resizing
- Train/validation/test split creation
- EfficientNet-B3 model setup
- Two-stage transfer learning
- Model training
- Saving final model artifacts

## 2. CosmicVision_2_0.ipynb

This notebook contains the final evaluation pipeline. It includes:

- Loading the trained EfficientNet-B3 model
- Recreating the test dataset setup
- Final test-set evaluation
- Classification report
- Confusion matrix
- Normalized confusion matrix
- Grad-CAM visualization outputs
- Final performance metrics

The second notebook should be referred to for detailed evaluation outputs and visual results.
