# House Price ML Workspace

This workspace is organized for teaching and experimentation.

## Project Layout

- `ml101_basic/`
  - Beginner-friendly Random Forest project for class demos.
- `legacy/pytorch_house_price/`
  - Original PyTorch neural-network workflow and assets.
- `legacy/archive/`
  - Older backup/experimental copies kept for reference.
- `examples/`
  - Place new future examples here.

## Quick Start (ML 101)

1. Go to `ml101_basic/`
2. Train:

```powershell
python train.py
```

3. Predict:

```powershell
python predict.py --zipcode 94102 --house_sqft 1500 --lot_sqft 4000 --num_beds 3 --num_baths 2.0 --property_type 1
```

## Notes

- ML101 trained artifacts are stored in `ml101_basic/models/`.
- Legacy PyTorch artifacts are in `legacy/pytorch_house_price/models/`.
- Keep adding new demos under `examples/` to avoid clutter at workspace root.
