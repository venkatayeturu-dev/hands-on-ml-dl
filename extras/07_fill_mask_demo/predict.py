"""
Fill-Mask Demo — How Transformers "Read" a Whole Sentence at Once

The model sees ALL tokens in parallel and predicts the missing word.
This is impossible with left-to-right (RNN) models.
"""

from transformers import pipeline

# Load a pre-trained BERT model for fill-mask
unmasker = pipeline("fill-mask", model="bert-base-uncased")

# Try different sentences — [MASK] is the word the model predicts
sentences = [
    "The capital of France is [MASK].",
    "She went to the [MASK] to buy groceries.",
    "Water freezes at zero degrees [MASK].",
    "The cat sat on the [MASK] and fell asleep.",
]

print("=" * 60)
print("FILL-MASK DEMO — Transformer reads ALL words at once")
print("=" * 60)

for sentence in sentences:
    print(f"\nInput: {sentence}")
    results = unmasker(sentence)
    print("Top predictions:")
    for r in results[:3]:
        print(f"  {r['token_str']:>12}  ({r['score']:.1%})")
