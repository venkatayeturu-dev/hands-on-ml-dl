# Hands-On ML & DL Primer

A focused primer on Machine Learning and Deep Learning fundamentals — designed as a prerequisite before students move to the **Agentic AI** track (RAG, MCP, local LLMs, agents).

> **Goal:** Demystify ML/DL in 1-2 sessions so students understand what models are, how training works, and why they'll be *consumers* of pre-trained LLMs — not trainers.

---

## Course Path (Recommended Order)

| # | Folder | What it teaches |
|---|--------|-----------------|
| 00 | `00_ml101_basic/` | Classic ML regression (Random Forest) — house price prediction |
| 01 | `01_ml101_classification_demo/` | Classic ML classification (Logistic Regression) — spam detection |
| 03 | `03_ml101_image_classification_demo/` | Deep Learning CNN — cat vs dog image classification |
| 05 | `05_inference_only_demo/` | **Using a pre-trained model** — no training, just inference (the bridge to Agentic AI) |

### Optional Deep-Dives

| # | Folder | What it teaches |
|---|--------|-----------------|
| 02 | `extras/02_ml101_pytorch_house_price/` | Same regression problem but in PyTorch (ML vs DL side-by-side) |
| 04 | `extras/04_ml10_image_defect_detection_demo/` | CNN for manufacturing defect detection (real-world applied DL) |

---

## Quick Start

```powershell
# Activate virtual environment
.\.venv\Scripts\Activate.ps1   # or: .\env\Scripts\Activate.ps1

# Run any project
cd 00_ml101_basic
python train.py
python predict.py --zipcode 94102 --house_sqft 1500 --lot_sqft 4000 --num_beds 3 --num_baths 2.0 --property_type 1
```

---

## Documentation

| Doc | Purpose |
|-----|---------|
| `docs/landscape.md` | Where ML, DL, LLMs, and Agentic AI fit — the big picture |
| `docs/training_process.md` | The 9-step ML/DL workflow with worked examples |
| `docs/addtional_info.md` | Quick-reference: epochs, batches, loss functions |
| `docs/summary.md` | Cheat sheet: what you learned and what comes next |

---

## Key Takeaway for Agentic AI Students

You will **not** train LLMs from scratch. You will:
- Use pre-trained models (Ollama, OpenAI, HuggingFace)
- Call inference APIs
- Build agents that chain LLM calls together
- Provide context via RAG and MCP

This repo teaches you what's *inside* those models so you understand what you're working with.
