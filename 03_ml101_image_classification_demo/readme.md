# ML 101 Deep Learning Demo: Cat vs Dog (CNN)

This demo teaches image classification with a simple Convolutional Neural Network (CNN).

## What Students Learn

- What a CNN is (basic idea)
- How deep learning training differs from classical ML
- How to train an image classifier
- How to predict class and confidence for one image

## Dataset

- Source: CIFAR-10 (auto-downloaded by `train.py`)
- We only keep two classes:
  - `cat`
  - `dog`

## Files

- `train.py` - downloads data, trains CNN, saves model + metrics
- `predict.py` - predicts cat/dog for an image path
- `models/cat_dog_cnn.pt` - trained model checkpoint
- `models/metrics.json` - training summary and best test accuracy
- `samples/sample_cat.png` - saved sample image after training
- `samples/sample_dog.png` - saved sample image after training

## Train

```powershell
python train.py
```

Quick smoke test:

```powershell
python train.py --epochs 1 --max_train_samples 2000 --max_test_samples 400
```

## Predict

CLI mode:

```powershell
python predict.py --image_path samples/sample_cat.png
python predict.py --image_path samples/sample_dog.png
```

Interactive mode:

```powershell
python predict.py
```

Then enter an image path when prompted.

## High-Level Training Flow

1. Download CIFAR-10
2. Filter data to cat and dog classes only
3. Convert images to tensors and normalize
4. Train CNN for multiple epochs
5. Evaluate on test split each epoch
6. Save best model checkpoint and metrics

## Why CNN (not R-CNN/YOLO) for ML 101?

- This task is **classification** (one label per image).
- CNN is the simplest deep-learning model for this use case.
- R-CNN/YOLO are object detection methods (more advanced, for bounding boxes).
