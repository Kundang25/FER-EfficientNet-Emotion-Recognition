
# Facial Emotion Recognition using EfficientNetB2

A Deep Learning based Computer Vision project for Facial Emotion Recognition using Transfer Learning with EfficientNetB2 on the FER2013 dataset.  
The model is trained using multi-phase fine-tuning, data augmentation, class balancing, and Test Time Augmentation to improve generalization performance.

---

## Project Overview

Facial Emotion Recognition (FER) is a Computer Vision task that aims to automatically identify human emotions from facial images.  
This project uses a pretrained EfficientNetB2 model and transfer learning to classify facial expressions into seven emotion categories.

The model is trained on the FER2013 dataset and evaluated using test accuracy, confusion matrix, and classification report.

---

## Emotion Classes

The model classifies images into the following emotions:

- Angry
- Disgust
- Fear
- Happy
- Neutral
- Sad
- Surprise

---

## Dataset

**Dataset Used:** FER2013 (Kaggle)

- 48x48 grayscale facial images
- 7 emotion classes
- ~35,000 images
- Train / Validation / Test split
- Imbalanced dataset

Dataset Link:  
https://www.kaggle.com/datasets/msambare/fer2013

---

## Model Architecture

The model uses Transfer Learning with EfficientNetB2 as the feature extractor.

**Architecture:**
```

Input Image (224x224x3)
↓
Rescaling Layer
↓
Data Augmentation
↓
EfficientNetB2 (Pretrained on ImageNet)
↓
Global Average Pooling
↓
Batch Normalization
↓
Dense Layer (512)
↓
Dropout
↓
Dense Layer (256)
↓
Dropout
↓
Output Layer (Softmax - 7 Classes)

```

---

## Training Strategy

The model was trained in **three phases**:

| Phase | Description |
|------|-------------|
| Phase 1 | Train classification head (backbone frozen) |
| Phase 2 | Unfreeze top layers and fine-tune |
| Phase 3 | Unfreeze full model and fine-tune with very low learning rate |

This multi-phase training improves generalization and prevents overfitting.

---

## Techniques Used

The following deep learning techniques were used:

- Transfer Learning
- EfficientNetB2
- Data Augmentation
- Label Smoothing
- Class Weights (for imbalanced dataset)
- Dropout Regularization
- Batch Normalization
- AdamW Optimizer
- Learning Rate Scheduling
- Early Stopping
- Model Checkpointing
- Test Time Augmentation (TTA)
- Confusion Matrix
- Classification Report

---

## Results

| Metric | Accuracy |
|-------|----------|
| Training Accuracy | ~92% |
| Validation Accuracy | 95.03% |
| Test Accuracy | 68.96% |
| Test Accuracy (TTA) | 69.59% |

Test Time Augmentation improved the model performance slightly by averaging predictions over augmented test images.

---

## Confusion Matrix

The confusion matrix shows that the model performs best on **Happy** and **Surprise** emotions, while **Fear** and **Sad** are harder to classify due to similar facial features.

(Add confusion matrix image here if you saved one)

---

## Project Structure

```

FER-EfficientNet-Emotion-Recognition/
│
├── notebook/
│   └── fer_training.ipynb
│
├── model/
│   └── best_emotion_model.keras
│
├── app/
│   └── app.py
│
├── images/
│   └── confusion_matrix.png
│
├── requirements.txt
├── README.md
└── .gitignore

```

---

## Installation

Clone the repository:

```

git clone [https://github.com/yourusername/FER-EfficientNet-Emotion-Recognition.git](https://github.com/yourusername/FER-EfficientNet-Emotion-Recognition.git)
cd FER-EfficientNet-Emotion-Recognition

```

Install dependencies:

```

pip install -r requirements.txt

```

---

## How to Run

### Train Model
Run the Jupyter Notebook:
```

fer_training.ipynb

```

### Run Flask App
```

python app.py

```

Open browser:
```

[http://127.0.0.1:5000/](http://127.0.0.1:5000/)

```

Upload a face image to detect emotion.

---

## Applications

- Human Computer Interaction
- Emotion Aware Systems
- Mental Health Monitoring
- Smart Surveillance
- Customer Behavior Analysis
- Social Robotics

---

## Future Improvements

- Use Attention Mechanisms
- Use Vision Transformers
- Face Landmark Detection
- Model Quantization for Edge Devices
- Real-time Video Emotion Detection
- Ensemble Models

---

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn
- Flask
- Jupyter Notebook

---

## Author

**Kundan Gupta**  
B.Tech Computer Science  
Computer Vision / Deep Learning Project

---

## License

This project is for educational and research purposes.



