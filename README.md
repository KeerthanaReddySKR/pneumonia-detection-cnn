# Pneumonia Detection from Chest X-ray Images using Deep Learning

## Overview

This project presents a deep learning-based system for automated detection of pneumonia from chest X-ray images. Leveraging transfer learning with a pretrained convolutional neural network, the model is designed to distinguish between normal and pneumonia cases with reliable performance.

The workflow follows a structured and reproducible machine learning pipeline, including data preprocessing, model development, training, and evaluation.
## Workflow Overview

```mermaid
flowchart TD

A[Dataset: Chest X-ray Images] --> B[Data Preprocessing]
B --> B1[Resize Images (224x224)]
B --> B2[Normalize Pixel Values]
B --> B3[Data Augmentation]

B3 --> C[Data Loading]
C --> C1[Train Generator]
C --> C2[Validation Generator]
C --> C3[Test Generator]

C1 --> D[Model Development]
D --> D1[Load ResNet50 (Pretrained)]
D --> D2[Freeze Base Layers]
D --> D3[Add Custom Classification Head]

D3 --> E[Model Training]
E --> E1[Train on Training Data]
E --> E2[Validate on Validation Data]
E --> E3[Early Stopping & Checkpointing]

E3 --> F[Model Evaluation]
F --> F1[Accuracy]
F --> F2[Precision, Recall, F1-score]
F --> F3[Confusion Matrix]

F3 --> G[Results & Visualization]
G --> G1[Accuracy Graph]
G --> G2[Performance Analysis]
```

---

## Problem Statement

Pneumonia is a potentially life-threatening respiratory condition that requires timely and accurate diagnosis. Chest X-ray imaging is widely used for detection; however, manual interpretation can be time-intensive and subject to variability.

This project addresses the problem by developing an automated classification system that assists in identifying pneumonia from X-ray images.

---

## Dataset

The dataset used is the **Chest X-Ray Images (Pneumonia)** dataset sourced from Kaggle. It consists of labeled X-ray images categorized into two classes:

* NORMAL
* PNEUMONIA

The dataset is organized into training, validation, and test subsets to support model development and unbiased evaluation.

```id="7j1n2h"
data/
└── chest_xray/
    ├── train/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    ├── val/
    │   ├── NORMAL/
    │   └── PNEUMONIA/
    └── test/
        ├── NORMAL/
        └── PNEUMONIA/
```

---

## Project Structure

```id="c0tq4k"
pneumonia-detection-cnn/
│
├── data/
│   └── chest_xray/
│
├── src/
│   ├── data_loader.py        # Data preprocessing and generators
│   ├── model.py              # Model architecture (ResNet50)
│   ├── train.py              # Training pipeline
│   ├── evaluate.py           # Evaluation and metrics
│
├── models/
│   └── best_model.h5         # Saved trained model
│
├── outputs/
│   └── accuracy.png          # Training visualization
│
├── requirements.txt
├── README.md
└── LICENSE
```

---

## Methodology

### Data Preprocessing

All images were resized to 224 × 224 pixels to match the input requirements of the pretrained model. Pixel intensities were normalized to the range [0, 1].

To improve generalization, data augmentation was applied to the training set using:

* Random rotations
* Zoom transformations
* Horizontal flipping

Validation and test datasets were kept unchanged to ensure unbiased evaluation.

---

### Data Pipeline

The dataset was loaded using TensorFlow/Keras `ImageDataGenerator`, enabling efficient batch-wise loading and automatic label assignment based on directory structure.

Separate generators were defined for training, validation, and testing.

---

### Model Architecture

Transfer learning was implemented using **ResNet50**, pretrained on ImageNet.

* The convolutional base was loaded with pretrained weights and without the top classification layer.
* Base layers were frozen to retain learned feature representations.
* A custom classification head was added:

  * Global Average Pooling layer
  * Fully connected Dense layer (ReLU activation)
  * Sigmoid output layer for binary classification

This approach enables effective feature extraction while adapting the model to the medical imaging domain.

---

### Training Strategy

The model was trained using:

* Binary Crossentropy loss
* Adam optimizer
* Accuracy as the evaluation metric

To improve training stability and prevent overfitting:

* Early stopping was applied based on validation loss
* Model checkpointing was used to save the best-performing model

---

### Evaluation

Model performance was evaluated on the test dataset using:

* Accuracy
* Precision
* Recall
* F1-score
* Confusion Matrix

These metrics provide a comprehensive assessment of classification performance.

---

## Results

The model achieved approximately **76% accuracy** on the test dataset.

Performance analysis shows:

* High recall for pneumonia detection (~90%), indicating strong ability to identify positive cases
* Moderate performance on normal class detection, influenced by dataset imbalance

The confusion matrix indicates that the model is more inclined toward predicting pneumonia, which is acceptable in clinical screening scenarios where minimizing false negatives is critical.

<img width="712" height="348" alt="Screenshot 2026-03-24 134919" src="https://github.com/user-attachments/assets/f6c2d015-1bb3-41f5-b837-57fe21f27de0" />

---

## Visualization

Training performance was visualized using accuracy curves across epochs. This helps in understanding model convergence and identifying potential overfitting.

The generated plot is available in the `outputs` directory.

---

## Key Insights

* Transfer learning significantly improves performance on limited medical datasets
* Data augmentation enhances generalization capability
* Class imbalance influences prediction bias
* High recall for pneumonia detection aligns with clinical priorities

---

## Limitations

* Very small validation dataset, leading to unstable validation metrics
* Class imbalance between normal and pneumonia images
* Lack of model interpretability techniques such as Grad-CAM

---

## Future Scope

* Fine-tuning deeper layers of the pretrained model
* Expanding validation dataset for better performance estimation
* Incorporating explainable AI techniques
* Extending the model to multi-class disease classification

---

## Reproducibility

To reproduce the results:

```id="9j6x1n"
git clone https://github.com/your-username/pneumonia-detection-cnn.git
cd pneumonia-detection-cnn
pip install -r requirements.txt
python src/train.py
python src/evaluate.py
```

---

## Conclusion

This project demonstrates the application of deep learning and transfer learning in medical image classification. The developed model provides a practical and scalable approach for pneumonia detection and highlights the potential of AI-assisted diagnostic systems in healthcare.

---

## Author

Keerthana Reddy
B.Tech Bioinformatics
