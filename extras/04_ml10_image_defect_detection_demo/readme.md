# ML 101 Deep Learning Demo: Image Defect Detection (Synthetic Data)

This demo teaches binary image classification for quality inspection.

## What Students Learn

- How to generate synthetic training data for DL
- How to train a CNN for defect vs good classification
- How to evaluate on validation and test sets
- How to run single-image predictions with confidence

## Dataset Strategy

This demo creates images locally with no external download.

- `good` images: textured clean surfaces
- `defect` images: clean surfaces with synthetic scratches, spots, and chipped patches

Generated folder structure:

- `data/train/good`, `data/train/defect`
- `data/val/good`, `data/val/defect`
- `data/test/good`, `data/test/defect`

## Files

- `generate_data.py` - creates synthetic dataset splits
- `train.py` - trains CNN and saves best model + metrics
- `predict.py` - predicts `GOOD` or `DEFECT` for one image
- `models/defect_cnn.pt` - model checkpoint
- `models/metrics.json` - training summary and final test metrics

## Step 1: Generate Data

```powershell
python generate_data.py --recreate
```

Quick smaller dataset for smoke tests:

```powershell
python generate_data.py --recreate --train_per_class 120 --val_per_class 40 --test_per_class 40
```

## Step 2: Train

```powershell
python train.py
```

Quick smoke test training:

```powershell
python train.py --epochs 1 --batch_size 32
```

## Step 3: Predict

```powershell
python predict.py --image_path data/test/good/good_0000.png
python predict.py --image_path data/test/defect/defect_0000.png
```

Interactive mode:

```powershell
python predict.py
```

## Notes

- This is a teaching demo, not a production inspection model.
- Replace synthetic data with real camera images using the same folder structure to adapt it to your process.
