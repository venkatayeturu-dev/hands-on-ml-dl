"""
06: Inference Only — Translation (No Training!)

This script demonstrates using a pre-trained translation model.
Same concept as 05 (sentiment) but with a different task: English → Spanish.

There is no train.py here. The model was trained by the Helsinki NLP group
on millions of sentence pairs. You just download and use it.

Run:
    python predict.py --text "The weather is beautiful today"
    python predict.py --text "I need to buy groceries after work"
    python predict.py   (interactive mode)
"""

import argparse
from transformers import MarianMTModel, MarianTokenizer


def load_model():
    """Load pre-trained English-to-Spanish translation model.

    First run downloads ~300 MB. Subsequent runs use the cache.
    No GPU required — runs on CPU.
    """
    print("Loading pre-trained translation model (first run downloads ~300 MB)...")

    # Model: Helsinki-NLP/opus-mt-en-es
    # Source: https://huggingface.co/Helsinki-NLP/opus-mt-en-es
    #
    # How it works:
    #   - On first run, HuggingFace downloads the model weights from their servers.
    #   - Files are cached locally at: ~/.cache/huggingface/hub/
    #   - Subsequent runs load instantly from cache (no internet needed).
    #
    # What the model is:
    #   - MarianMT = a Transformer-based seq2seq (sequence-to-sequence) model.
    #   - Trained by Helsinki NLP group on OPUS parallel corpus data.
    #   - Input: English text → Output: Spanish translation.
    #   - This is the same Transformer architecture that powers LLMs,
    #     but smaller and specialized for one task (translation).
    #
    # Connection to Agentic AI:
    #   - LLMs (GPT, LLaMA) can also translate — but they're general-purpose.
    #   - This model is task-specific: smaller, faster, but only does EN→ES.
    #   - In Agentic AI, you'd use an LLM for translation as part of a larger workflow.
    model_name = "Helsinki-NLP/opus-mt-en-es"
    tokenizer = MarianTokenizer.from_pretrained(model_name)
    model = MarianMTModel.from_pretrained(model_name)

    print("Model ready.\n")
    return tokenizer, model


def translate(translator, text):
    """Run inference on a single text input."""
    tokenizer, model = translator
    inputs = tokenizer(text, return_tensors="pt")
    outputs = model.generate(**inputs)
    translation = tokenizer.decode(outputs[0], skip_special_tokens=True)

    print(f"  English: {text}")
    print(f"  Spanish: {translation}")
    print()
    return translation


def main():
    parser = argparse.ArgumentParser(
        description="English to Spanish translation using a pre-trained model (no training needed)"
    )
    parser.add_argument("--text", type=str, default=None, help="English text to translate")
    args = parser.parse_args()

    model = load_model()

    if args.text:
        translate(model, args.text)
    else:
        # Interactive mode
        print("=" * 50)
        print("TRANSLATION DEMO (English → Spanish)")
        print("Type any English sentence to get a translation.")
        print("Type 'quit' to exit.")
        print("=" * 50)
        print()

        while True:
            text = input("English: ").strip()
            if text.lower() in ("quit", "exit", "q"):
                break
            if not text:
                continue
            translate(model, text)


if __name__ == "__main__":
    main()
