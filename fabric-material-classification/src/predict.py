"""Run top-k predictions on one image."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import BinaryIO

import torch
from PIL import Image

from src.dataset import CLASSES, build_transforms
from src.model import build_model


def load_predictor(checkpoint_path: Path, device=None):
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {checkpoint_path}. Train the model first.")
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    classes = checkpoint.get("classes", CLASSES)
    model = build_model(len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    _, transform = build_transforms(checkpoint.get("image_size", 224))
    return model, classes, transform, device


def predict_image(image_path: Path | BinaryIO, model, classes, transform, device, top_k: int = 3):
    with Image.open(image_path) as image:
        tensor = transform(image.convert("RGB")).unsqueeze(0).to(device)
    with torch.no_grad():
        probabilities = torch.softmax(model(tensor), dim=1)[0]
    k = min(max(1, top_k), len(classes))
    values, indices = probabilities.topk(k)
    return [(classes[index], float(value)) for value, index in zip(values.cpu(), indices.cpu())], probabilities.cpu().tolist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("image", type=Path)
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_model.pth"))
    parser.add_argument("--top-k", type=int, default=3)
    args = parser.parse_args()
    if not args.image.is_file():
        raise FileNotFoundError(f"Image not found: {args.image}")
    model, classes, transform, device = load_predictor(args.checkpoint)
    top, probabilities = predict_image(args.image, model, classes, transform, device, args.top_k)
    print(f"Prediction: {top[0][0]} | Confidence: {top[0][1] * 100:.2f}%")
    print("Top predictions:")
    for name, probability in top:
        print(f"  {name}: {probability * 100:.2f}%")
    print("All class probabilities:")
    for name, probability in zip(classes, probabilities):
        print(f"  {name}: {probability * 100:.2f}%")


if __name__ == "__main__":
    main()

