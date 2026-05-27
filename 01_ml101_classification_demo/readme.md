# ML 101 Classification Demo (Spam Detection)

This is a simple beginner project that teaches how classification works using Logistic Regression.

## What This Demo Teaches

- How to generate a dataset
- How to train a basic classifier
- How to evaluate classification metrics
- How to run predictions with new input values

## Problem Statement

Predict whether a message is:

- `1` = Spam
- `0` = Not spam

using these features:

- `num_links`
- `num_exclamations`
- `has_urgent_word` (0 or 1)
- `uppercase_ratio` (0.0 to 1.0)
- `message_length`

## Files

- `generate_data.py` - creates synthetic dataset (500 rows)
- `train.py` - trains Logistic Regression and saves model
- `predict.py` - predicts spam/not spam (CLI + interactive)
- `data/spam_data.csv` - generated dataset
- `models/spam_logreg.pkl` - trained model
- `models/scaler.pkl` - feature scaler
- `models/metrics.json` - accuracy, precision, recall, confusion matrix

## Step 1: Generate Data

```powershell
python generate_data.py
```

## Step 2: Train Model

```powershell
python train.py
```

## Step 3: Predict

### Option A: Command-line input

```powershell
python predict.py --num_links 3 --num_exclamations 4 --has_urgent_word 1 --uppercase_ratio 0.6 --message_length 80
```

Another sample:

```powershell
python predict.py --num_links 0 --num_exclamations 0 --has_urgent_word 0 --uppercase_ratio 0.1 --message_length 240
```

### Option B: Interactive mode

```powershell
python predict.py
```

## How Training Works (Simple)

1. Load generated CSV data
2. Split into train/test sets
3. Scale features (important for Logistic Regression)
4. Train classifier on train set
5. Evaluate with:
   - Accuracy
   - Precision
   - Recall
   - Confusion Matrix
6. Save model and scaler

## Metric Meanings

- Accuracy: overall percent of correct predictions
- Precision: when model says spam, how often that is correct
- Recall: of all real spam messages, how many the model found
- Confusion Matrix: counts of correct and wrong predictions by class
