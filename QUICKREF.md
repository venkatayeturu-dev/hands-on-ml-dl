# Quick Reference — Run Commands

All commands assume you're in the repo root with the virtual environment activated.

---

## Core Path

### Demo 00 — ML Regression (House Price)

```powershell
cd 00_ml101_basic
python train.py
python predict.py --zipcode 94102 --house_sqft 1500 --lot_sqft 4000 --num_beds 3 --num_baths 2.0 --property_type 1
```

### Demo 01 — ML Classification (Spam Detection)

```powershell
cd 01_ml101_classification_demo
python generate_data.py
python train.py
python predict.py --num_links 3 --num_exclamations 4 --has_urgent_word 1 --uppercase_ratio 0.6 --message_length 80
```

### Demo 03 — Deep Learning CNN (Cat vs Dog)

```powershell
cd 03_ml101_image_classification_demo

# Quick smoke test (2 minutes)
python train.py --epochs 1 --max_train_samples 2000 --max_test_samples 400

# Full training (10-20 minutes on CPU)
python train.py

# Predict
python predict.py --image_path samples/sample_cat.png
python predict.py --image_path samples/sample_dog.png
```

### Demo 05 — Inference Only (Sentiment Analysis)

```powershell
cd 05_inference_only_demo
python predict.py --text "I love this product, it works great!"
python predict.py --text "This is terrible, complete waste of money."
```

---

## Optional Extras

### Demo 02 — PyTorch Regression (same problem as Demo 00)

```powershell
cd extras/02_ml101_pytorch_house_price
python train.py
python predict.py --zipcode 94102 --house_sqft 1500 --lot_sqft 4000 --num_beds 3 --num_baths 2.0 --property_type 1
```

### Demo 04 — Defect Detection CNN

```powershell
cd extras/04_ml10_image_defect_detection_demo
python generate_data.py
python train.py --epochs 1 --max_train_samples 500
python predict.py --image_path data/test/sample_defect.png
```

### Demo 06 — Translation (English → Spanish)

```powershell
cd extras/06_inference_translation_demo
python predict.py --text "The weather is beautiful today"
```

### Demo 07 — Fill Mask (BERT)

```powershell
cd extras/07_fill_mask_demo
python predict.py
```
