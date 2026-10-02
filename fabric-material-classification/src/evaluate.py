"""Evaluate a saved checkpoint once on the held-out test manifest."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report, confusion_matrix
from torch.utils.data import DataLoader

from src.dataset import CLASSES, FabricDataset, build_transforms
from src.model import build_model


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("manifests/test.csv"))
    parser.add_argument("--checkpoint", type=Path, default=Path("models/best_model.pth"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--batch-size", type=int, default=32)
    args = parser.parse_args()
    if not args.checkpoint.exists():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}. Run training first.")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    classes = checkpoint.get("classes", CLASSES)
    if classes != CLASSES:
        raise ValueError(f"Checkpoint classes do not match expected order: {classes}")
    _, transform = build_transforms(checkpoint.get("image_size", 224))
    dataset = FabricDataset(args.manifest, transform)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(len(classes), pretrained=False).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    targets, predictions = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images.to(device))
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
            targets.extend(labels.tolist())
    args.outputs.mkdir(parents=True, exist_ok=True)
    report = classification_report(targets, predictions, labels=list(range(len(classes))), target_names=classes, digits=4, zero_division=0)
    accuracy = accuracy_score(targets, predictions)
    (args.outputs / "classification_report.txt").write_text(
        f"Test accuracy: {accuracy:.4f}\n\n{report}\n", encoding="utf-8"
    )
    matrix = confusion_matrix(targets, predictions, labels=list(range(len(classes))))
    fig, ax = plt.subplots(figsize=(9, 8))
    ConfusionMatrixDisplay(matrix, display_labels=classes).plot(ax=ax, cmap="Blues", xticks_rotation=35, colorbar=False)
    ax.set_title("Test Confusion Matrix")
    fig.tight_layout()
    fig.savefig(args.outputs / "confusion_matrix.png", dpi=160)
    plt.close(fig)
    print(f"Test accuracy: {accuracy:.4f}\n\n{report}")
    print(f"Saved report and confusion matrix to {args.outputs}")


if __name__ == "__main__":
    main()

