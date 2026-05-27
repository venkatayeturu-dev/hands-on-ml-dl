# ML 101 House Price Predictor (Random Forest)

This folder contains a very basic machine learning project for predicting house prices.

## High-Level Overview

- Goal: Predict `price_usd` for a house.
- Model: `RandomForestRegressor` (from scikit-learn).
- Input features:
  - `market_rate_per_sqft` (looked up from zipcode)
  - `house_sqft`
  - `lot_sqft`
  - `num_beds`
  - `num_baths`
  - `property_type` (`1` = Single Family, `2` = Condo/Multifamily)

## Files in This Folder

- `train.py` - Trains the Random Forest model.
- `predict.py` - Uses the trained model to predict price.
- `house_data.csv` - Training dataset.
- `zip_rates.json` - Zipcode to market-rate mapping used in features.
- `models/house_price_rf.pkl` - Saved trained model (created after training).
- `models/metrics_rf.json` - Saved model metrics (created after training).

## How Training Works (Simple Explanation)

1. Load data from `house_data.csv`.
2. Select feature columns (`X`) and target column (`y` = `price_usd`).
3. Split data into training and test sets (80/20).
4. Train a Random Forest with `model.fit(X_train, y_train)`.
5. Evaluate on test data with:
   - RMSE (error in dollars)
   - R2 score (how much variance is explained)
   - MAE (average absolute error)
6. Save outputs to the `models` folder.

### Important Note About Epochs

Random Forest does **not** use epochs (that is common in neural networks).
It builds many decision trees in one `.fit()` call.

## How to Train

Run from this folder:

```powershell
python train.py
```

Expected outputs:

- `models/house_price_rf.pkl`
- `models/metrics_rf.json`

## How to Run Prediction

You can run prediction in 2 ways.

### Option 1: Command-Line Arguments

```powershell
python predict.py --zipcode 94102 --house_sqft 1500 --lot_sqft 4000 --num_beds 3 --num_baths 2.0 --property_type 1
```

Another sample:

```powershell
python predict.py --zipcode 10003 --house_sqft 900 --lot_sqft 0 --num_beds 2 --num_baths 1.5 --property_type 2
```

### Option 2: Interactive Mode

```powershell
python predict.py
```

Then enter values when prompted.

## Typical Classroom Flow

1. Run `python train.py`
2. Check metrics in `models/metrics_rf.json`
3. Run `python predict.py` with sample inputs
4. Change inputs and observe prediction behavior

## MAE and RMSE (What They Mean)

- MAE (Mean Absolute Error):
  - This is the average absolute prediction error.
  - If MAE is $60,000, predictions are off by about $60,000 on average.

- RMSE (Root Mean Squared Error):
  - This also measures prediction error, but it penalizes large mistakes more.
  - If RMSE is much higher than MAE, the model likely has some big misses.

How to read them together:

- Lower MAE and lower RMSE are better.
- MAE = "average miss".
- RMSE = "average miss, with extra penalty for large errors".
