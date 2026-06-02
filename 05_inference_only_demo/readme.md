# 05 — Inference Only Demo (The Bridge to Agentic AI)

This demo has **no training step**. You use a pre-trained model to run predictions.

## The Point

In every previous project, you ran `train.py` first, then `predict.py`.
Here, there is **no `train.py`**. Someone else already trained the model.
You just download it and use it.

> This is exactly how you'll work with LLMs in Agentic AI —
> you download a pre-trained model and call it.

## What This Demo Does

Uses a pre-trained sentiment analysis model from HuggingFace to classify text as POSITIVE or NEGATIVE.

## Install Extra Dependency

```powershell
pip install transformers
```

> Note: This uses the `transformers` library from HuggingFace. The first run
> downloads a small pre-trained model (~260 MB). No GPU needed.

## Run

```powershell
python predict.py --text "I love this product, it works great!"
python predict.py --text "This is terrible, complete waste of money."
```

Interactive mode:

```powershell
python predict.py
```

## Key Observations for Students

1. **No `train.py` exists** — you didn't train anything.
2. **The model was trained by someone else** — on thousands of GPUs, with millions of text samples.
3. **You just call it** — input text → model → output label + confidence.
4. **This is the same pattern you'll use with LLMs** — except LLMs generate text instead of labels.

## Connection to Agentic AI

| This demo | Agentic AI |
|-----------|-----------|
| `pipeline("sentiment-analysis")` | `ollama.chat(model="llama3")` |
| Downloads a pre-trained model | Downloads a pre-trained LLM |
| Input: text → Output: label | Input: prompt → Output: generated text |
| One model, one task | One LLM, many tasks (via prompting) |
| ~260 MB model | ~4-8 GB model |
