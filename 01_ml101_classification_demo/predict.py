"""
ML 101: Spam Prediction (Logistic Regression)

Examples:
    python predict.py
    python predict.py --num_links 3 --num_exclamations 4 --has_urgent_word 1 --uppercase_ratio 0.6 --message_length 80
"""

import argparse
import pickle  # SECURITY: Only load pickle files you created yourself – pickle can execute arbitrary code.
from pathlib import Path

import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "spam_logreg.pkl"
SCALER_FILE = BASE_DIR / "models" / "scaler.pkl"

FEATURE_COLS = [
    "num_links",
    "num_exclamations",
    "has_urgent_word",
    "uppercase_ratio",
    "message_length",
]


def load_files():
    if not MODEL_FILE.exists() or not SCALER_FILE.exists():
        raise FileNotFoundError("Model files not found. Run: python train.py")

    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)

    with open(SCALER_FILE, "rb") as f:
        scaler = pickle.load(f)

    return model, scaler


def parse_args():
    parser = argparse.ArgumentParser(description="Predict spam/not-spam")
    parser.add_argument("--num_links", type=int)
    parser.add_argument("--num_exclamations", type=int)
    parser.add_argument("--has_urgent_word", type=int, choices=[0, 1])
    parser.add_argument("--uppercase_ratio", type=float)
    parser.add_argument("--message_length", type=int)
    return parser.parse_args()


def interactive_input():
    print("Enter message feature values:")
    num_links = int(input("Number of links (0-8): ").strip())
    num_exclamations = int(input("Number of exclamation marks (0-10): ").strip())
    has_urgent_word = int(input("Contains urgent words? (0=no, 1=yes): ").strip())
    uppercase_ratio = float(input("Uppercase ratio (0.0 to 1.0): ").strip())
    message_length = int(input("Message length (characters): ").strip())

    return num_links, num_exclamations, has_urgent_word, uppercase_ratio, message_length


def main():
    args = parse_args()

    all_args_given = all(
        value is not None
        for value in [
            args.num_links,
            args.num_exclamations,
            args.has_urgent_word,
            args.uppercase_ratio,
            args.message_length,
        ]
    )

    if all_args_given:
        values = (
            args.num_links,
            args.num_exclamations,
            args.has_urgent_word,
            args.uppercase_ratio,
            args.message_length,
        )
    else:
        values = interactive_input()

    try:
        model, scaler = load_files()
    except FileNotFoundError as e:
        print(e)
        return

    row = pd.DataFrame([values], columns=FEATURE_COLS)
    row_s = scaler.transform(row)

    pred = int(model.predict(row_s)[0])
    prob_spam = float(model.predict_proba(row_s)[0][1])

    label = "SPAM" if pred == 1 else "NOT SPAM"

    print("\nInput features:")
    print(row.to_dict(orient="records")[0])
    print(f"\nPrediction: {label}")
    print(f"Spam probability: {prob_spam:.1%}")


if __name__ == "__main__":
    main()
