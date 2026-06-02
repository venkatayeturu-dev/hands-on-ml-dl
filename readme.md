# Hands-On ML & DL Primer

### AI/ML Foundations for Enterprise Practitioners

**Instructor:** Vaithi Ramadoss (vaithi.ramadoss@gmail.com)  
**Format:** Self-paced, ~5-6 hours total  
**Prerequisite for:** [Enterprise-Grade Agentic AI Course](https://github.com/vramados/enterprise-agentic-ai)

---

> **Goal:** Demystify ML/DL so you understand what models are, how training works, and why you'll be *consumers* of pre-trained LLMs — not trainers.

---

## Learning Path

Complete these in order (~5-6 hours total):

```
+-----------------------------------------------------------------+
|                                                                   |
|   00 ML Regression --> 01 ML Classification --> 03 DL (CNN)       |
|   (1 hr)               (1 hr)                  (1.5 hr)          |
|                                                                   |
|                                         |                         |
|                                         v                         |
|                                                                   |
|                                   05 Inference Only --> DONE      |
|                                   (45 min)        |               |
|                                                   v               |
|                                                                   |
|                             Start Agentic AI Course               |
|                                                                   |
+-----------------------------------------------------------------+
```

| # | Folder | What You Learn | Time |
|---|--------|----------------|:----:|
| 00 | `00_ml101_basic/` | ML regression - train a house price predictor (Random Forest) | 1 hr |
| 01 | `01_ml101_classification_demo/` | ML classification - build a spam detector (Logistic Regression) | 1 hr |
| 03 | `03_ml101_image_classification_demo/` | Deep Learning - train a CNN image classifier (cat vs dog) | 1.5 hr |
| 05 | `05_inference_only_demo/` | **Inference only** - use a pre-trained model with no training step | 45 min |

> **Why the numbering gaps?** Demos 02 and 04 are optional deep-dives in `extras/`. The core path skips them.

### Optional Deep-Dives (extras/)

| # | Folder | What You Learn | Time |
|---|--------|----------------|:----:|
| 02 | `extras/02_ml101_pytorch_house_price/` | Same regression problem in PyTorch (ML vs DL comparison) | 1 hr |
| 04 | `extras/04_ml10_image_defect_detection_demo/` | CNN for manufacturing defect detection (real-world DL) | 1.5 hr |
| 06 | `extras/06_inference_translation_demo/` | Pre-trained translation model (English to Spanish) | 30 min |
| 07 | `extras/07_fill_mask_demo/` | Masked language model (fill-in-the-blank) | 20 min |

---

## The Key Insight

```
+-----------------------------------------------------------------+
|                                                                   |
|   In this repo:    YOU train small models from scratch            |
|   In Agentic AI:   YOU consume pre-trained LLMs (no training!)   |
|                                                                   |
|   Demo 05 is the bridge between these two worlds.                 |
|                                                                   |
+-----------------------------------------------------------------+
```

After completing Demo 05, you understand:
- What a model **is** (learned weights that map input -> output)
- What **inference** is (using a model without training it)
- Why you **don't train LLMs** yourself (someone else spent months + millions on GPU clusters)

You're ready for the [Agentic AI course](https://github.com/vramados/enterprise-agentic-ai).

---

## Documentation (Read in Order)

| # | Doc | What It Covers |
|---|-----|----------------|
| 1 | [docs/01-landscape.md](docs/01-landscape.md) | The full AI stack: ML -> DL -> LLMs -> Agents (with Mermaid diagram) |
| 2 | [docs/02-training-process.md](docs/02-training-process.md) | The 9-step ML/DL workflow, worked examples, key vocabulary |
| 3 | [docs/03-summary.md](docs/03-summary.md) | Cheat sheet: what you learned + vocabulary bridge to Agentic AI |

---

## Quick Start

```powershell
# Clone
git clone https://github.com/vramados/hands-on-ml-dl.git
cd hands-on-ml-dl

# Setup (see PREREQUISITES.md for full details)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt

# Run your first demo
cd 00_ml101_basic
python train.py
python predict.py --zipcode 94102 --house_sqft 1500 --lot_sqft 4000 --num_beds 3 --num_baths 2.0 --property_type 1
```

---

## Academic Context

This primer is part of a curriculum inspired by:
- **UC Berkeley** - LLM Agents MOOC (CS 294)
- **Stanford** - CS 224N / CS 329
- **Johns Hopkins** - Agentic AI Certificate

We distill foundational concepts into a practitioner-focused format - no math prerequisites, 100% hands-on.

---

## What's Next

After completing this primer: [**Enterprise-Grade Agentic AI Course**](https://github.com/vramados/enterprise-agentic-ai) (12 weeks, instructor-led)

---

## License

This work is licensed under [CC BY-NC-SA 4.0](LICENSE).
Copyright 2026 Vaithi Ramadoss.
