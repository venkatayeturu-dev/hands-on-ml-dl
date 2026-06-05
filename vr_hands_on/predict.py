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
MODEL_FILE = BASE_DIR / "models" / "car_price_trends.pkl"



def load_files():
    with open(MODEL_FILE, "rb") as f:
        model = pickle.load(f)
    return model

#sale_price
def build_features(date,make,model,type,condition,mileage,list_price,days_on_lot,region,fuel_type,year):
   

   
    # Feature names and order must match train.py.
    return pd.DataFrame(
        [[date,make,model,type,condition,mileage,list_price,days_on_lot,region,fuel_type,year]],
        columns=[
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


def interactive_input():
    # print("Supported zipcodes:")
    # print(sorted(zip_rates.keys()))
    print()

#df = pd.read_csv(DATA_FILE, dtype={'date':int,'make':str, 'model':str, 'type':str, 'condition':str, 'mileage':int, 'list_price':float, 'sale_price':float, 
# 'days_on_lot':int, 'region':str, 'fuel_type':str, 'year':int})

    date = int(input("Date: ").strip())
    make = str(input("Make: " ).strip())
    model = str(input("Model: ").strip())
    type = str(input("Type: ").strip())
    condition = str(input("Condition: ").strip())
    mileage = int(input("Mileage: ").strip())
    list_price = float(input("List Price: ").strip())
    days_on_lot = int(input("Days on lot: ").strip())
    region = str(input("Region: ").strip())
    fuel_type = str(input("Fuel Type: ").strip())
    year = int(input("year: ").strip())

    return date, make, model, type, condition, mileage,list_price,days_on_lot,region,fuel_type,year


def main():
 #   args = parse_args()
    model = load_files()
    #print(type(model))
    # all_args_given = all(
    #     value is not None
    #     for value in [
    #         args.zipcode,
    #         args.house_sqft,
    #         args.lot_sqft,
    #         args.num_beds,
    #         args.num_baths,
    #         args.property_type,
    #     ]
    # )

#    if all_args_given:
#        zipcode = args.zipcode
#        house_sqft = args.house_sqft
#        lot_sqft = args.lot_sqft
#        num_beds = args.num_beds
#        num_baths = args.num_baths
#        property_type = args.property_type
#    else:
    date, make, model, type, condition, mileage,list_price,days_on_lot,region,fuel_type,year = interactive_input()

    try:
        X = build_features(
        date, make, model, type, condition, mileage,list_price,days_on_lot,region,fuel_type,year,
        )
    except ValueError as e:
        print(f"Error: {e}")
        return

    print("\nInput features:", X.to_dict(orient="records")[0])
    prediction = float(model.predict(X)[0])
    print(f"\nPredicted house price: ${prediction:,.0f}")


if __name__ == "__main__":
    main()
