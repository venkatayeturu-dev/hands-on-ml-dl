"""
Predict defect vs good for one image.

Examples:
    python predict.py --image_path data/test/good/good_0000.png
    python predict.py --image_path data/test/defect/defect_0000.png
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms


BASE_DIR = Path(__file__).resolve().parent
MODEL_FILE = BASE_DIR / "models" / "defect_cnn.pt"


class DefectCNN(nn.Module):
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
            nn.Linear(128 * 16 * 16, 128),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(128, 2),
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def parse_args():
    parser = argparse.ArgumentParser(description="Predict defect/good for one image")
    parser.add_argument("--image_path", type=str, default="")
    return parser.parse_args()


def load_model_and_meta():
    if not MODEL_FILE.exists():
        raise FileNotFoundError("Model not found. Run: python train.py")

    # SECURITY: weights_only=False can execute arbitrary code – only load trusted model files.
    checkpoint = torch.load(MODEL_FILE, map_location="cpu", weights_only=False)

    model = DefectCNN()
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    class_names = checkpoint.get("class_names", ["defect", "good"])
    image_size = int(checkpoint.get("image_size", 128))
    mean = checkpoint.get("normalize_mean", [0.5, 0.5, 0.5])
    std = checkpoint.get("normalize_std", [0.5, 0.5, 0.5])

    return model, class_names, image_size, mean, std


def preprocess_image(image_path, image_size, mean, std):
    tfm = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    img = Image.open(image_path).convert("RGB")
    x = tfm(img).unsqueeze(0)
    return x


def main():
    args = parse_args()
    image_path = args.image_path.strip()

    if not image_path:
        image_path = input("Enter image path: ").strip()

    if not image_path:
        print("No image path provided.")
        return

    image_file = Path(image_path)
    if not image_file.exists():
        print(f"Image not found: {image_file}")
        return

    try:
        model, class_names, image_size, mean, std = load_model_and_meta()
    except FileNotFoundError as e:
        print(e)
        return

    x = preprocess_image(image_file, image_size, mean, std)

    with torch.no_grad():
        logits = model(x)
        probs = torch.softmax(logits, dim=1)
        pred_idx = int(probs.argmax(dim=1).item())
        confidence = float(probs[0, pred_idx].item())

    print(f"Prediction: {class_names[pred_idx].upper()}")
    print(f"Confidence: {confidence:.1%}")


if __name__ == "__main__":
    main()
