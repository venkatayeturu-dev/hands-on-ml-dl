# 06 — Inference Only: Translation Demo (English → Spanish)

This demo has **no training step**. You use a pre-trained translation model.

## The Point

Same concept as `05_inference_only_demo` but with a different task:
- 05 = sentiment classification (text → label)
- 06 = translation (English text → Spanish text)

Both use pre-trained Transformer models. Neither requires training.

## What This Demo Does

Uses a pre-trained MarianMT model from Helsinki NLP to translate English text to Spanish.

## Install Dependencies

```powershell
pip install transformers sentencepiece
```

> `sentencepiece` is required by the MarianMT tokenizer.

## Run

```powershell
python predict.py --text "The weather is beautiful today"
python predict.py --text "Machine learning models learn patterns from data"
```

Interactive mode:

```powershell
python predict.py
```

## Key Observations for Students

1. **Still no `train.py`** — you didn't train anything.
2. **Different task, same pattern** — download model, call it, get output.
3. **Sequence-to-sequence** — unlike sentiment (text → label), this is text → text.
4. **This IS what LLMs do** — LLMs are also seq2seq Transformers, just much bigger and general-purpose.

## Connection to Agentic AI

| This demo | LLM in Agentic AI |
|-----------|-------------------|
| Specialized model (EN→ES only) | General-purpose (translates any language pair) |
| ~300 MB | ~4-8 GB |
| One task | Many tasks via prompting |
| Deterministic output | Temperature-controlled output |
| `translator("Hello")` → Spanish | `llm("Translate to Spanish: Hello")` → Spanish |

**Key insight:** LLMs replaced many task-specific models by being general-purpose. Instead of downloading 50 specialized models, you use 1 LLM and prompt it differently each time.
