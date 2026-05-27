"""
ML 101: Spam Classification Training (Logistic Regression)

Run:
    python train.py
"""

import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "data" / "spam_data.csv"
MODELS_DIR = BASE_DIR / "models"
MODEL_FILE = MODELS_DIR / "spam_logreg.pkl"
SCALER_FILE = MODELS_DIR / "scaler.pkl"
METRICS_FILE = MODELS_DIR / "metrics.json"

FEATURE_COLS = [
    "num_links",
    "num_exclamations",
    "has_urgent_word",
    "uppercase_ratio",
    "message_length",
]
TARGET_COL = "is_spam"


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if not DATA_FILE.exists():
        print("Data file not found.")
        print("Run: python generate_data.py")
        return

    print("Loading data...")
    df = pd.read_csv(DATA_FILE)

    X = df[FEATURE_COLS]
    y = df[TARGET_COL]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    print("Training Logistic Regression...")
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train_s, y_train)

    preds = model.predict(X_test_s)

    acc = float(accuracy_score(y_test, preds))
    prec = float(precision_score(y_test, preds, zero_division=0))
    rec = float(recall_score(y_test, preds, zero_division=0))
    cm = confusion_matrix(y_test, preds).tolist()

    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model, f)

    with open(SCALER_FILE, "wb") as f:
        pickle.dump(scaler, f)

    metrics = {
        "model": "LogisticRegression",
        "accuracy": round(acc, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "confusion_matrix": cm,
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "feature_cols": FEATURE_COLS,
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    print("Done.")
    print(f"Model saved: models/{MODEL_FILE.name}")
    print(f"Scaler saved: models/{SCALER_FILE.name}")
    print(f"Metrics saved: models/{METRICS_FILE.name}")
    print(f"Accuracy: {acc:.3f} | Precision: {prec:.3f} | Recall: {rec:.3f}")


if __name__ == "__main__":
    main()
