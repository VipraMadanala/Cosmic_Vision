# Sequence-Based Classification of Astronomical Objects Using Deep Learning

**Project Name:** CosmicVision  
**Event:** Summer Siege × Odyssey 2026  
**Institution:** IIT Gandhinagar  

---

## Overview

CosmicVision is an astronomical image classification web application built using EfficientNet-B3 with Grad-CAM explainability.

The application allows users to upload an astronomical image and predicts one of five celestial object categories:

- Elliptical Galaxy
- Nebula
- Planetary Object
- Spiral Galaxy
- Star Cluster

Along with the predicted class, the dashboard displays confidence scores for all classes and a Grad-CAM visualization showing the image regions used by the model for prediction.

---

## How to Run

Clone or download this repository.

Install the required dependencies:

```bash
pip install -r requirements.txt
```

Run the dashboard:

```bash
python app.py
```
After the app starts, open the local URL shown in the terminal, usually:

```text
http://127.0.0.1:7860
```

Then upload an astronomical image and click **Predict**.

---

## Final Performance

The final model achieved the following results:

| Metric | Value |
|---|---:|
| Test Accuracy | 93.51% |
| Macro F1-score | 90.62% |
| Weighted F1-score | 93.52% |

---

**Vipra Madanala**  
**B.Tech AI 2025**
