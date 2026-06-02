# Prerequisites & Setup

Everything you need before running the demos in this repo.

---

## System Requirements

| Requirement | Minimum | Recommended |
|-------------|---------|-------------|
| **OS** | Windows 10, macOS 12, Ubuntu 22.04 | Windows 11, macOS 14 |
| **Python** | 3.11 | 3.11 (not 3.12+ due to torch compatibility) |
| **RAM** | 8 GB | 16 GB (for CNN training in Demo 03) |
| **Disk** | 2 GB free | 5 GB free (model downloads) |
| **GPU** | NOT required | NOT required (all demos run on CPU) |
| **Internet** | Required for Demos 05-07 | Required (model downloads) |

---

## Step-by-Step Setup

### 1. Install Python 3.11

- **Windows:** Download from https://www.python.org/downloads/release/python-3119/  
  > ⚠️ Check "Add Python to PATH" during installation
- **macOS:** `brew install python@3.11`
- **Linux:** `sudo apt install python3.11 python3.11-venv`

### 2. Clone the Repository

```powershell
git clone https://github.com/vramados/hands-on-ml-dl.git
cd hands-on-ml-dl
```

### 3. Create Virtual Environment

```powershell
# Windows
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Mac/Linux
python3.11 -m venv .venv
source .venv/bin/activate
```

> **Windows Execution Policy Error?** Run this first:
> ```powershell
> Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned
> ```

### 4. Install Dependencies

```powershell
pip install -r requirements.txt
```

### 5. Verify Installation

```powershell
python -c "import torch; import sklearn; import numpy; import pandas; print('All packages installed successfully!')"
```

---

## What Gets Downloaded Automatically

Some demos download pre-trained models on first run:

| Demo | What Downloads | Size | When |
|------|---------------|------|------|
| 03 (CNN) | CIFAR-10 dataset | ~170 MB | First `python train.py` |
| 05 (Inference) | HuggingFace sentiment model | ~260 MB | First `python predict.py` |
| extras/06 (Translation) | MarianMT model | ~300 MB | First `python predict.py` |
| extras/07 (Fill Mask) | BERT model | ~440 MB | First `python predict.py` |

These are cached locally after the first download.

---

## Troubleshooting

| Issue | Fix |
|-------|-----|
| `python` not found | Reinstall Python with "Add to PATH" checked |
| `pip install torch` fails | Try: `pip install torch --index-url https://download.pytorch.org/whl/cpu` |
| Execution policy error (Windows) | `Set-ExecutionPolicy -Scope Process -ExecutionPolicy RemoteSigned` |
| Out of memory during CNN training | Use: `python train.py --max_train_samples 2000 --epochs 1` |
| Slow first run on Demos 05-07 | Normal — models are downloading. Subsequent runs are instant. |

---

## Need Help?

Contact: vaithi.ramadoss@gmail.com
