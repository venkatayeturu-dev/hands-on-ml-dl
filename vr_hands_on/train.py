"""
ML 101: House Price Training (Random Forest)

Run:
    python train.py

Training overview (very simple):
1) Load the CSV data.
2) Choose input columns (features) and output column (target).
3) Split data into train/test sets.
4) Fit a Random Forest model on the training set.
5) Evaluate on the test set (RMSE, R2, MAE) and save files.

Note:
- Random Forest does not train with epochs.
- It builds many decision trees in one fit() call.
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "car_price_trends1.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "car_price_trends.pkl"
METRICS_FILE = MODELS_DIR / "car_price_trends.json"

# Keep features simple and explicit for students.
FEATURE_COLS = [
"date",
"make",
"model",
"type",
"condition",
"mileage",
"list_price",
"days_on_lot",
"region",
"fuel_type",
"year",

]
TARGET_COL = "sale_price"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    print("Training overview:")
    print("1) Load data")
    print("2) Build features and target")
    print("3) Split into train/test")
    print("4) Train Random Forest (no epochs)")
    print("5) Evaluate and save model")
    print()

    print("Loading data...")
    #df = pd.read_csv(DATA_FILE, parse_dates=['date'])
#    df = pd.read_csv(DATA_FILE)
#
     
 
    le = LabelEncoder()
 
    # nrwos = 2
    df = pd.read_csv(DATA_FILE, dtype={'date':int,'make':str, 'model':str, 'type':str, 'condition':str, 'mileage':int, 'list_price':float, 'sale_price':float, 'days_on_lot':int, 'region':str, 'fuel_type':str, 'year':int})

    for col in ['make', 'model', 'type', 'condition', 'region', 'fuel_type']:
    
        df[col] = le.fit_transform(df[col])
    

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    # Standard 80/20 split for train/test.

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    print("Training Random Forest...")
    # n_estimators=200 means we build 200 trees.
    model = RandomForestRegressor(
        n_estimators=200,
        random_state=42,
        n_jobs=-1,
    )
    
    print("before model fit...")
    model.fit(X_train, y_train)
    print("after model fit...")
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))
    mae = float(np.mean(np.abs(y_test - preds)))

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    metrics = {
        "model": "RandomForestRegressor",
        "rmse_usd": round(rmse),
        "r2_score": round(r2, 4),
        "mae_usd": round(mae),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_cols": FEATURE_COLS,
    }
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Done.")
    print(f"Model saved: models/{MODEL_FILE.name}")
    print(f"Metrics saved: models/{METRICS_FILE.name}")
    print(f"RMSE: ${rmse:,.0f} | R2: {r2:.4f} | MAE: ${mae:,.0f}")


if __name__ == "__main__":
   main()
