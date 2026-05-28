"""
05: Inference Only — Using a Pre-Trained Model (No Training!)

This script demonstrates the mental shift from ML/DL (where you train)
to Agentic AI (where you consume pre-trained models).

There is no train.py in this folder. The model was trained by someone else.
You just download and use it.

Run:
    python predict.py --text "I love this!"
    python predict.py --text "This is awful."
    python predict.py   (interactive mode)
"""

import argparse
from transformers import pipeline


def load_model():
    """Load pre-trained sentiment analysis model.

    First run downloads ~260 MB. Subsequent runs use the cache.
    No GPU required — runs on CPU in under a second.
    """
    print("Loading pre-trained model (first run downloads ~260 MB)...")
    classifier = pipeline("sentiment-analysis")
    print("Model ready.\n")
    return classifier


def predict(classifier, text):
    """Run inference on a single text input."""
    result = classifier(text)[0]
    label = result["label"]
    score = result["score"]

    print(f"  Input:      {text}")
    print(f"  Prediction: {label}")
    print(f"  Confidence: {score:.4f}")
    print()
    return result


def main():
    parser = argparse.ArgumentParser(
        description="Sentiment analysis using a pre-trained model (no training needed)"
    )
    parser.add_argument("--text", type=str, default=None, help="Text to analyze")
    args = parser.parse_args()

    classifier = load_model()

    if args.text:
        predict(classifier, args.text)
    else:
        # Interactive mode
        print("=" * 50)
        print("INFERENCE-ONLY DEMO")
        print("Type any sentence to get sentiment prediction.")
        print("Type 'quit' to exit.")
        print("=" * 50)
        print()

        while True:
            text = input("Enter text: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            predict(classifier, text)


if __name__ == "__main__":
    main()
