# Advanced Neural Networks for CIFAR-10

This notebook explores different neural network architectures for image classification on the **CIFAR-10 dataset**. It includes multiple implementations, progressing from simpler models to an advanced **Convolutional Neural Network (CNN)** achieving at least **85% accuracy**.

## Contents

- **Basic CNN Model**: An initial convolutional network with a simpler structure.
- **Advanced CNN Model**: A deeper architecture with batch normalization, dropout, and fully connected layers.
- **Prediction Functions**:
  - `predict()`: Standard inference for the test dataset.
  - `predict_tta()`: Test Time Augmentation (TTA) to improve prediction stability by averaging augmented test results.
- **Training & Evaluation**: Functions to train and evaluate the models.
- **Saving & Loading Models**: Utilizing `torch.save` to store model weights and predictions.

## Model Architectures

### 1. Basic Convolutional Neural Network
A straightforward CNN with fewer layers to classify CIFAR-10 images.

### 2. Advanced Convolutional Neural Network
A deeper architecture consisting of:
- **Three convolutional blocks** with increasing channels: `32 → 64 → 128`
- **Batch normalization** after each convolutional layer
- **ReLU activation functions** for better feature extraction
- **Max pooling layers** for downsampling
- **Dropout layers** for regularization
- **Fully connected layers** with a final softmax output

## Requirements

Install dependencies with:

```bash
pip install torch torchvision matplotlib numpy
