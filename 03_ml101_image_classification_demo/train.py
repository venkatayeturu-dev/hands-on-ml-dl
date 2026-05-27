"""
ML 101: Cat vs Dog Image Classification (PyTorch CNN)

This script downloads CIFAR-10, keeps only cat/dog images,
trains a simple CNN, evaluates it, and saves a model.

Default run (better accuracy mode):
    python train.py

Quick smoke test run:
    python train.py --epochs 1 --max_train_samples 2000 --max_test_samples 400
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from PIL import Image


BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
SAMPLES_DIR = BASE_DIR / "samples"
MODEL_FILE = MODELS_DIR / "cat_dog_cnn.pt"
METRICS_FILE = MODELS_DIR / "metrics.json"

# CIFAR-10 labels: airplane=0, automobile=1, bird=2, cat=3, deer=4,
#                 dog=5, frog=6, horse=7, ship=8, truck=9
CAT_LABEL = 3
DOG_LABEL = 5
CLASS_NAMES = ["cat", "dog"]


class CatDogCIFAR(Dataset):
    """Filter CIFAR-10 to cat/dog and map labels to 0/1."""

    def __init__(self, root, train, transform, download, max_samples=0):
        self.base = datasets.CIFAR10(root=str(root), train=train, transform=transform, download=download)

        allowed = {CAT_LABEL: 0, DOG_LABEL: 1}
        self.indices = [i for i, t in enumerate(self.base.targets) if t in allowed]

        if max_samples and max_samples > 0:
            self.indices = self.indices[:max_samples]

        self.label_map = allowed

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        base_idx = self.indices[idx]
        image, label = self.base[base_idx]
        return image, self.label_map[label]


class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def train_one_epoch(model, loader, optimizer, loss_fn, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = loss_fn(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = logits.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total


def evaluate(model, loader, loss_fn, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            logits = model(images)
            loss = loss_fn(logits, labels)

            running_loss += loss.item() * images.size(0)
            preds = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total


def save_sample_images(root_dir):
    """Save one cat and one dog sample image for quick prediction demos."""
    raw = datasets.CIFAR10(root=str(root_dir), train=False, download=True)

    found_cat = False
    found_dog = False

    for img_array, label in zip(raw.data, raw.targets):
        if label == CAT_LABEL and not found_cat:
            Image.fromarray(img_array).save(SAMPLES_DIR / "sample_cat.png")
            found_cat = True
        elif label == DOG_LABEL and not found_dog:
            Image.fromarray(img_array).save(SAMPLES_DIR / "sample_dog.png")
            found_dog = True

        if found_cat and found_dog:
            break


def parse_args():
    parser = argparse.ArgumentParser(description="Train CNN for cat vs dog classification")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--max_train_samples", type=int, default=0)
    parser.add_argument("--max_test_samples", type=int, default=0)
    return parser.parse_args()


def main():
    args = parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLES_DIR.mkdir(parents=True, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ]
    )

    print("Loading CIFAR-10 and filtering cat/dog...")
    train_ds = CatDogCIFAR(
        root=DATA_DIR,
        train=True,
        transform=transform,
        download=True,
        max_samples=args.max_train_samples,
    )
    test_ds = CatDogCIFAR(
        root=DATA_DIR,
        train=False,
        transform=transform,
        download=True,
        max_samples=args.max_test_samples,
    )

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False)

    print(f"Train images: {len(train_ds)}")
    print(f"Test images : {len(test_ds)}")

    model = SimpleCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_test_acc = 0.0

    print("\nTraining...")
    for epoch in range(1, args.epochs + 1):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, loss_fn, device)
        test_loss, test_acc = evaluate(model, test_loader, loss_fn, device)

        print(
            f"Epoch {epoch:02d}/{args.epochs} | "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.3f} | "
            f"test_loss={test_loss:.4f} test_acc={test_acc:.3f}"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "class_names": CLASS_NAMES,
                    "input_size": [3, 32, 32],
                    "normalize_mean": [0.4914, 0.4822, 0.4465],
                    "normalize_std": [0.2470, 0.2435, 0.2616],
                },
                MODEL_FILE,
            )

    metrics = {
        "model": "SimpleCNN",
        "dataset": "CIFAR-10 filtered to cat/dog",
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.lr,
        "train_images": len(train_ds),
        "test_images": len(test_ds),
        "best_test_accuracy": round(best_test_acc, 4),
        "device": str(device),
    }

    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    save_sample_images(DATA_DIR)

    print("\nDone.")
    print(f"Model saved  : models/{MODEL_FILE.name}")
    print(f"Metrics saved: models/{METRICS_FILE.name}")
    print("Sample images: samples/sample_cat.png, samples/sample_dog.png")


if __name__ == "__main__":
    main()
