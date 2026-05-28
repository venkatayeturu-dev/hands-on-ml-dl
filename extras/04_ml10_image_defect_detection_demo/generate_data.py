"""
Generate synthetic image defect dataset for binary classification.

Creates dataset folders:
- data/train/good
- data/train/defect
- data/val/good
- data/val/defect
- data/test/good
- data/test/defect

Run:
    python generate_data.py

Quick custom run:
    python generate_data.py --image_size 128 --train_per_class 300 --val_per_class 80 --test_per_class 80
"""

import argparse
import random
import shutil
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
SPLITS = ["train", "val", "test"]
CLASSES = ["good", "defect"]


def parse_args():
    parser = argparse.ArgumentParser(description="Generate synthetic defect image dataset")
    parser.add_argument("--image_size", type=int, default=128)
    parser.add_argument("--train_per_class", type=int, default=600)
    parser.add_argument("--val_per_class", type=int, default=150)
    parser.add_argument("--test_per_class", type=int, default=150)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--recreate", action="store_true", help="Delete existing data directory first")
    return parser.parse_args()


def make_base_texture(image_size, rng):
    # Build a subtle industrial-like texture from gradients + noise.
    y = np.linspace(0, 1, image_size, dtype=np.float32)
    x = np.linspace(0, 1, image_size, dtype=np.float32)
    grid_x, grid_y = np.meshgrid(x, y)

    gradient = 100 + 80 * grid_x + 30 * np.sin(6 * np.pi * grid_y)
    noise = rng.normal(0, 12, size=(image_size, image_size))
    img = np.clip(gradient + noise, 0, 255).astype(np.uint8)

    rgb = np.stack([img, img, img], axis=-1)
    pil = Image.fromarray(rgb).filter(ImageFilter.GaussianBlur(radius=0.6))
    return pil


def add_defects(img, rng):
    draw = ImageDraw.Draw(img)
    w, h = img.size

    # Scratches.
    scratch_count = rng.integers(1, 4)
    for _ in range(scratch_count):
        x1, y1 = int(rng.integers(0, w)), int(rng.integers(0, h))
        x2 = int(np.clip(x1 + rng.integers(-w // 3, w // 3), 0, w - 1))
        y2 = int(np.clip(y1 + rng.integers(-h // 3, h // 3), 0, h - 1))
        thickness = int(rng.integers(1, 4))
        shade = int(rng.integers(10, 70))
        draw.line((x1, y1, x2, y2), fill=(shade, shade, shade), width=thickness)

    # Spots.
    spot_count = rng.integers(1, 5)
    for _ in range(spot_count):
        cx, cy = int(rng.integers(0, w)), int(rng.integers(0, h))
        r = int(rng.integers(4, 14))
        shade = int(rng.integers(150, 240))
        draw.ellipse((cx - r, cy - r, cx + r, cy + r), fill=(shade, shade, shade))

    # Chipped patch.
    if rng.random() < 0.6:
        x1 = int(rng.integers(0, w - 20))
        y1 = int(rng.integers(0, h - 20))
        x2 = int(np.clip(x1 + rng.integers(10, 30), 0, w - 1))
        y2 = int(np.clip(y1 + rng.integers(10, 30), 0, h - 1))
        shade = int(rng.integers(20, 235))
        draw.rectangle((x1, y1, x2, y2), fill=(shade, shade, shade))

    return img.filter(ImageFilter.GaussianBlur(radius=0.4))


def save_split(split_name, per_class, image_size, seed):
    rng = np.random.default_rng(seed)
    for class_name in CLASSES:
        out_dir = DATA_DIR / split_name / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        for i in range(per_class):
            base = make_base_texture(image_size, rng)
            if class_name == "defect":
                base = add_defects(base, rng)

            file_name = f"{class_name}_{i:04d}.png"
            base.save(out_dir / file_name)


def main():
    args = parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    if args.recreate and DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)

    split_counts = {
        "train": args.train_per_class,
        "val": args.val_per_class,
        "test": args.test_per_class,
    }

    for idx, split_name in enumerate(SPLITS):
        split_seed = args.seed + idx * 1000
        save_split(split_name, split_counts[split_name], args.image_size, split_seed)

    print("Done generating synthetic dataset.")
    print(f"Image size: {args.image_size}x{args.image_size}")
    for split_name in SPLITS:
        count = split_counts[split_name]
        print(f"{split_name}: good={count}, defect={count}, total={count * 2}")


if __name__ == "__main__":
    main()
