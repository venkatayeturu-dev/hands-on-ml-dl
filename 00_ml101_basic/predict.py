"""
ML 101: House Price Prediction (Random Forest)

Examples:
    python predict.py
    python predict.py --zipcode 94102 --house_sqft 1500 --lot_sqft 4000 --num_beds 3 --num_baths 2.0 --property_type 1
"""

import argparse
import json
import pickle  # SECURITY: Only load pickle files you created yourself – pickle can execute arbitrary code.
from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "house_price_rf.pkl"
ZIP_FILE = BASE_DIR / "zip_rates.json"


def load_files():
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    with open(ZIP_FILE, "r", encoding="utf-8") as f:
        zip_rates = {int(k): v for k, v in json.load(f).items()}
    return model, zip_rates


def build_features(zipcode, house_sqft, lot_sqft, num_beds, num_baths, property_type, zip_rates):
    if zipcode not in zip_rates:
        raise ValueError(f"Zipcode {zipcode} is not supported.")

    market_rate = zip_rates[zipcode]

    # Feature names and order must match train.py.
    return pd.DataFrame(
        [[market_rate, house_sqft, lot_sqft, num_beds, num_baths, property_type]],
        columns=[
            "market_rate_per_sqft",
            "house_sqft",
            "lot_sqft",
            "num_beds",
            "num_baths",
            "property_type",
        ],
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Predict house price using Random Forest")
    parser.add_argument("--zipcode", type=int)
    parser.add_argument("--house_sqft", type=float)
    parser.add_argument("--lot_sqft", type=float)
    parser.add_argument("--num_beds", type=int)
    parser.add_argument("--num_baths", type=float)
    parser.add_argument("--property_type", type=int, choices=[1, 2])
    return parser.parse_args()


def interactive_input(zip_rates):
    print("Supported zipcodes:")
    print(sorted(zip_rates.keys()))
    print()

    zipcode = int(input("Zipcode: ").strip())
    house_sqft = float(input("House sqft: ").strip())
    lot_sqft = float(input("Lot sqft (0 for condo): ").strip())
    num_beds = int(input("Number of beds: ").strip())
    num_baths = float(input("Number of baths: ").strip())
    property_type = int(input("Property type (1=Single Family, 2=Condo/Multi): ").strip())

    return zipcode, house_sqft, lot_sqft, num_beds, num_baths, property_type


def main():
    args = parse_args()
    model, zip_rates = load_files()

    all_args_given = all(
        value is not None
        for value in [
            args.zipcode,
            args.house_sqft,
            args.lot_sqft,
            args.num_beds,
            args.num_baths,
            args.property_type,
        ]
    )

    if all_args_given:
        zipcode = args.zipcode
        house_sqft = args.house_sqft
        lot_sqft = args.lot_sqft
        num_beds = args.num_beds
        num_baths = args.num_baths
        property_type = args.property_type
    else:
        zipcode, house_sqft, lot_sqft, num_beds, num_baths, property_type = interactive_input(zip_rates)

    try:
        X = build_features(
            zipcode,
            house_sqft,
            lot_sqft,
            num_beds,
            num_baths,
            property_type,
            zip_rates,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    print("\nInput features:", X.to_dict(orient="records")[0])
    prediction = float(model.predict(X)[0])
    print(f"\nPredicted house price: ${prediction:,.0f}")


if __name__ == "__main__":
    main()
