"""Train ResNet18 and save the checkpoint selected by validation macro F1."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader

from src.dataset import CLASSES, FabricDataset, build_transforms, prepare_manifests, resolve_data_root, seed_everything
from src.model import build_model


def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    losses, targets, predictions = [], [], []
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            losses.append(criterion(logits, labels).item())
            targets.extend(labels.cpu().tolist())
            predictions.extend(logits.argmax(dim=1).cpu().tolist())
    return float(np.mean(losses)), accuracy_score(targets, predictions), f1_score(
        targets, predictions, average="macro", zero_division=0
    )


def save_training_curve(history, output_path: Path) -> None:
    epochs = [int(row["epoch"]) for row in history]
    train_losses = [float(row["train_loss"]) for row in history]
    val_losses = [float(row["validation_loss"]) for row in history]
    train_accuracies = [float(row["train_accuracy"]) for row in history]
    val_accuracies = [float(row["validation_accuracy"]) for row in history]
    val_macro_f1 = [float(row["validation_macro_f1"]) for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, train_losses, label="Train")
    axes[0].plot(epochs, val_losses, label="Validation")
    axes[0].set(title="Loss", xlabel="Epoch", ylabel="Cross-entropy")
    axes[1].plot(epochs, train_accuracies, label="Train accuracy")
    axes[1].plot(epochs, val_accuracies, label="Validation accuracy")
    axes[1].plot(epochs, val_macro_f1, label="Validation macro F1")
    axes[1].set(title="Metrics", xlabel="Epoch", ylabel="Score", ylim=(0, 1))
    for axis in axes:
        axis.legend()
        axis.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("dataset/raw"))
    parser.add_argument("--manifests", type=Path, default=Path("manifests"))
    parser.add_argument("--models", type=Path, default=Path("models"))
    parser.add_argument("--outputs", type=Path, default=Path("outputs"))
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--num-threads", type=int, default=4, help="Limit CPU thread count to keep training responsive.")
    parser.add_argument("--fine-tune", action="store_true", help="Fine-tune ResNet18's final block and classifier.")
    parser.add_argument("--resume-checkpoint", type=Path, help="Initialize from a previously trained checkpoint.")
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.num_threads < 1:
        parser.error("--epochs, --batch-size, and --num-threads must be positive")
    if args.learning_rate is None:
        args.learning_rate = 1e-5 if args.fine_tune else 1e-3
    torch.set_num_threads(args.num_threads)
    torch.set_num_interop_threads(1)
    seed_everything(args.seed)
    data_root = resolve_data_root(args.data)
    prepare_manifests(data_root, args.manifests, args.seed)
    train_transform, eval_transform = build_transforms()
    train_set = FabricDataset(args.manifests / "train.csv", train_transform)
    val_set = FabricDataset(args.manifests / "validation.csv", eval_transform)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, num_workers=0)
    model = build_model(len(CLASSES), pretrained=args.resume_checkpoint is None, freeze_backbone=True).to(device)
    resume_state = None
    if args.resume_checkpoint is not None:
        if not args.resume_checkpoint.exists():
            raise FileNotFoundError(f"Resume checkpoint not found: {args.resume_checkpoint}")
        resume_state = torch.load(args.resume_checkpoint, map_location=device, weights_only=False)
        if resume_state.get("classes", CLASSES) != CLASSES:
            raise ValueError("Resume checkpoint class order does not match this project.")
        model.load_state_dict(resume_state["model_state_dict"])
    if args.fine_tune:
        for parameter in model.parameters():
            parameter.requires_grad = False
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=args.learning_rate, weight_decay=1e-4
    )
    args.models.mkdir(parents=True, exist_ok=True)
    args.outputs.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.models / "best_model.pth"
    history_path = args.outputs / "training_history.csv"
    history = []
    if resume_state is not None and history_path.exists():
        with history_path.open(newline="", encoding="utf-8-sig") as handle:
            history = list(csv.DictReader(handle))
        for row in history:
            row["epoch"] = int(row["epoch"])
            row.setdefault("phase", "head")
    best_f1 = float(resume_state.get("best_validation_macro_f1", -1.0)) if resume_state else -1.0
    best_epoch = int(resume_state.get("epoch", 0)) if resume_state else 0
    best_phase = resume_state.get("phase", "head") if resume_state else None
    stale_epochs = 0
    started = time.time()
    print(f"Device: {device} | CPU threads={args.num_threads} | train={len(train_set)} | validation={len(val_set)}", flush=True)

    for epoch in range(1, args.epochs + 1):
        model.eval()
        if args.fine_tune:
            model.layer4.train()
        model.fc.train()
        losses, targets, predictions = [], [], []
        for batch_index, (images, labels) in enumerate(train_loader, start=1):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            targets.extend(labels.detach().cpu().tolist())
            predictions.extend(logits.detach().argmax(dim=1).cpu().tolist())
            if batch_index % 40 == 0 or batch_index == len(train_loader):
                print(f"Epoch {epoch:02d}/{args.epochs} | batch {batch_index}/{len(train_loader)}", flush=True)
        train_loss = float(np.mean(losses))
        train_accuracy = accuracy_score(targets, predictions)
        val_loss, val_accuracy, val_macro_f1 = evaluate_epoch(model, val_loader, criterion, device)
        global_epoch = (history[-1]["epoch"] if history else 0) + 1
        row = {
            "epoch": global_epoch,
            "phase": "fine-tune" if args.fine_tune else "head",
            "train_loss": train_loss,
            "train_accuracy": train_accuracy,
            "validation_loss": val_loss,
            "validation_accuracy": val_accuracy,
            "validation_macro_f1": val_macro_f1,
        }
        history.append(row)
        print(
            f"{row['phase']} {epoch:02d}/{args.epochs} | train loss {train_loss:.4f} acc {train_accuracy:.4f}"
            f" | val loss {val_loss:.4f} acc {val_accuracy:.4f} macro-F1 {val_macro_f1:.4f}",
            flush=True,
        )
        if val_macro_f1 > best_f1:
            best_f1, stale_epochs = val_macro_f1, 0
            best_epoch, best_phase = global_epoch, row["phase"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "image_size": 224,
                    "seed": args.seed,
                    "best_validation_macro_f1": best_f1,
                    "epoch": global_epoch,
                    "pretrained": True,
                    "fine_tune": args.fine_tune,
                    "phase": row["phase"],
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= args.patience:
                print(f"Early stopping: validation macro-F1 did not improve for {args.patience} epochs.")
                break

    with history_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=history[0].keys())
        writer.writeheader()
        writer.writerows(history)
    metadata = {
        "seed": args.seed,
        "device": str(device),
        "epochs_completed": len(history),
        "best_validation_macro_f1": best_f1,
        "best_epoch": best_epoch,
        "best_phase": best_phase,
        "best_checkpoint": str(checkpoint_path),
        "elapsed_seconds": round(time.time() - started, 2),
        "fine_tune": args.fine_tune,
    }
    (args.outputs / "training_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    save_training_curve(history, args.outputs / "training_curve.png")
    print(f"Saved best model: {checkpoint_path}")


if __name__ == "__main__":
    main()

