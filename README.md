# Pneumonia Detection from Chest X-ray Images using Deep Learning

## Overview
Deep learning-based system for classifying chest X-ray images into **Pneumonia** and **Normal** using transfer learning (ResNet50). The pipeline covers preprocessing, training, and evaluation with reproducible structure.

---

## Dataset
Chest X-Ray Images (Pneumonia) – Kaggle  
Classes: NORMAL, PNEUMONIA  

Structure:
data/
└── chest_xray/
    ├── train/
    ├── val/
    └── test/

---

## Tech Stack
Python, TensorFlow/Keras, NumPy, Pandas, Matplotlib, OpenCV, Scikit-learn

---

## Project Structure
pneumonia-detection-cnn/
├── data/
├── src/
│   ├── data_loader.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
├── models/
│   └── best_model.h5
├── outputs/
│   └── accuracy.png
├── requirements.txt
└── README.md

---

## Method

Preprocessing:
- Resize: 224×224  
- Normalize: 1/255  
- Augmentation: rotation, zoom, flip  

Model:
- ResNet50 (ImageNet pretrained)  
- Frozen base layers  
- Dense classifier (sigmoid output)  

Training:
- Loss: Binary Crossentropy  
- Optimizer: Adam  
- Early stopping + checkpointing  

---

## Workflow
```mermaid
flowchart TD
A[Dataset] --> B[Preprocessing]
B --> C[Data Generators]
C --> D[ResNet50 Model]
D --> E[Training]
E --> F[Evaluation]
F --> G[Results]
