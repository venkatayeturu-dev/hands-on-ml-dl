"""
ML 101: Generate synthetic spam classification data.

Run:
    python generate_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "spam_data.csv"
ROWS = 500


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(42)

    num_links = rng.integers(0, 9, size=ROWS)
    num_exclamations = rng.integers(0, 11, size=ROWS)
    has_urgent_word = rng.integers(0, 2, size=ROWS)
    uppercase_ratio = rng.uniform(0.0, 1.0, size=ROWS)
    message_length = rng.integers(20, 401, size=ROWS)

    # Simple synthetic rule: more links, urgency words, ALL CAPS, and many ! increase spam chance.
    logit = (
        -3.0
        + 0.55 * num_links
        + 0.30 * num_exclamations
        + 1.8 * has_urgent_word
        + 2.5 * uppercase_ratio
        - 0.004 * message_length
    )
    spam_prob = sigmoid(logit)
    is_spam = rng.binomial(1, spam_prob)

    df = pd.DataFrame(
        {
            "num_links": num_links,
            "num_exclamations": num_exclamations,
            "has_urgent_word": has_urgent_word,
            "uppercase_ratio": uppercase_ratio,
            "message_length": message_length,
            "is_spam": is_spam,
        }
    )

    df.to_csv(DATA_FILE, index=False)

    spam_rate = df["is_spam"].mean()
    print(f"Generated: {DATA_FILE}")
    print(f"Rows: {len(df)}")
    print(f"Spam rate: {spam_rate:.2%}")


if __name__ == "__main__":
    main()
