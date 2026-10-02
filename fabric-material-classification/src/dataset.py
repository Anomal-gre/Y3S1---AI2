"""Dataset inspection, leakage-aware splitting, and image transforms."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset
from torchvision import transforms

CLASSES = ["Cotton", "Cotton Mixed", "Denim", "Polyester", "Silk", "Viscose", "Wool"]
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _normalize_class_name(name: str) -> str:
    return " ".join(name.casefold().split())


def _class_directories(parent: Path) -> dict[str, Path]:
    if not parent.is_dir():
        return {}
    return {
        _normalize_class_name(child.name): child
        for child in parent.iterdir()
        if child.is_dir()
    }


def _official_split_dirs(path: Path) -> dict[str, Path] | None:
    if not path.is_dir():
        return None
    dirs = {child.name.casefold(): child for child in path.iterdir() if child.is_dir()}
    valid_dir = dirs.get("valid") or dirs.get("validation")
    if not all(key in dirs for key in ("train", "test")) or valid_dir is None:
        return None
    result = {"train": dirs["train"], "validation": valid_dir, "test": dirs["test"]}
    if any(not all(_normalize_class_name(name) in _class_directories(folder) for name in CLASSES) for folder in result.values()):
        return None
    return result


def resolve_data_root(path: Path) -> Path:
    """Accept flat class folders or an existing train/valid/test dataset export."""
    path = path.expanduser().resolve()
    candidates = [path, path / "raw", path / "dataset" / "raw", path.parent, path.parent.parent]
    for candidate in candidates:
        class_dirs = _class_directories(candidate)
        if all(_normalize_class_name(name) in class_dirs for name in CLASSES) or _official_split_dirs(candidate):
            return candidate
    expected = "\n".join(f"  {path / cls}" for cls in CLASSES)
    raise FileNotFoundError(
        f"Could not find all seven class folders under {path}. Expected folders:\n{expected}\n"
        "See dataset/raw/README.md for the required layout."
    )


def list_images(data_root: Path) -> list[dict[str, str]]:
    records: list[dict[str, str]] = []
    split_dirs = _official_split_dirs(data_root)
    sources = split_dirs.items() if split_dirs else [("", data_root)]
    for split, folder in sources:
        class_dirs = _class_directories(folder)
        for label, class_name in enumerate(CLASSES):
            class_folder = class_dirs[_normalize_class_name(class_name)]
            for file_path in sorted(class_folder.rglob("*")):
                if not file_path.is_file() or file_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                record = {"path": str(file_path.resolve()), "class_name": class_name, "label": str(label)}
                if split:
                    record["source_split"] = split
                record["source_id"] = re.sub(r"\.rf\..*$", "", file_path.stem, flags=re.IGNORECASE).casefold()
                records.append(record)
    return records


def inspect_images(data_root: Path) -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    """Decode images and report exact byte duplicates without inventing labels."""
    valid: list[dict[str, str]] = []
    invalid: list[dict[str, str]] = []
    hash_to_record: dict[str, dict[str, str]] = {}
    for record in list_images(data_root):
        path = Path(record["path"])
        try:
            with Image.open(path) as image:
                image.verify()
            digest = hashlib.sha256(path.read_bytes()).hexdigest()
            prior = hash_to_record.get(digest)
            record["sha256"] = digest
            record["duplicate_of"] = prior["path"] if prior else ""
            hash_to_record.setdefault(digest, record)
            valid.append(record)
        except Exception as exc:
            invalid.append({**record, "issue": f"Unreadable image: {exc}"})
    return valid, invalid


def _stratified_group_indices(records: list[dict[str, str]], seed: int) -> dict[str, list[dict[str, str]]]:
    """Split source-image families together, stratifying on their class labels."""
    groups: dict[str, list[dict[str, str]]] = {}
    for row in records:
        groups.setdefault(row["source_id"], []).append(row)
    group_ids = list(groups)
    group_labels = []
    for group_id in group_ids:
        labels = {row["label"] for row in groups[group_id]}
        if len(labels) != 1:
            raise ValueError(f"Source-image family has inconsistent labels: {group_id}")
        group_labels.append(next(iter(labels)))
    counts = Counter(group_labels)
    too_small = [CLASSES[int(label)] for label, count in counts.items() if count < 3]
    if too_small:
        raise ValueError(
            "Cannot make stratified train/validation/test splits with at least three unique image groups "
            f"per class. Insufficient classes: {', '.join(too_small)}"
        )

    train_val_ids, test_ids = train_test_split(
        group_ids, test_size=0.15, random_state=seed, stratify=group_labels
    )
    label_by_id = {group_id: group_labels[i] for i, group_id in enumerate(group_ids)}
    train_val_labels = [label_by_id[group_id] for group_id in train_val_ids]
    train_ids, val_ids = train_test_split(
        train_val_ids,
        test_size=0.15 / 0.85,
        random_state=seed,
        stratify=train_val_labels,
    )
    return {
        "train": [row for group_id in train_ids for row in groups[group_id]],
        "validation": [row for group_id in val_ids for row in groups[group_id]],
        "test": [row for group_id in test_ids for row in groups[group_id]],
    }


def prepare_manifests(data_root: Path, output_dir: Path, seed: int) -> None:
    valid, invalid = inspect_images(data_root)
    if not valid:
        raise ValueError(f"No readable images found in {data_root}")
    official_splits = _official_split_dirs(data_root)
    # Roboflow variants retain the source filename before `.rf.<hash>`. Quarantine a
    # whole family if it crosses a supplied split or appears with contradictory labels.
    source_groups: dict[str, list[dict[str, str]]] = {}
    hash_groups: dict[str, list[dict[str, str]]] = {}
    for row in valid:
        source_groups.setdefault(row["source_id"], []).append(row)
        hash_groups.setdefault(row["sha256"], []).append(row)
    quarantine_reasons: dict[str, set[str]] = {}
    for group in source_groups.values():
        reasons = set()
        if len({row["label"] for row in group}) > 1:
            reasons.add("source family has conflicting class labels")
        if official_splits and len({row.get("source_split") for row in group}) > 1:
            reasons.add("source family spans official data splits")
        if reasons:
            for row in group:
                quarantine_reasons[row["path"]] = reasons
    for group in hash_groups.values():
        reasons = set()
        if len({row["label"] for row in group}) > 1:
            reasons.add("identical file bytes have conflicting class labels")
        if official_splits and len({row.get("source_split") for row in group}) > 1:
            reasons.add("identical file bytes span official data splits")
        if reasons:
            for row in group:
                quarantine_reasons.setdefault(row["path"], set()).update(reasons)

    quarantined = [
        {**row, "quarantine_reason": "; ".join(sorted(quarantine_reasons[row["path"]]))}
        for row in valid
        if row["path"] in quarantine_reasons
    ]
    usable = [row for row in valid if row["path"] not in quarantine_reasons]
    if official_splits:
        split_records = {
            split: [row for row in usable if row.get("source_split") == split]
            for split in ("train", "validation", "test")
        }
    else:
        split_records = _stratified_group_indices(usable, seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    for split, records in split_records.items():
        manifest = output_dir / f"{split}.csv"
        import csv

        with manifest.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=["path", "class_name", "label", "sha256", "duplicate_of", "source_id"],
                extrasaction="ignore",
            )
            writer.writeheader()
            writer.writerows(records)
    import csv

    with (output_dir / "invalid_images.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["path", "class_name", "label", "issue"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(invalid)
    with (output_dir / "quarantined_images.csv").open("w", newline="", encoding="utf-8") as handle:
        fields = ["path", "class_name", "label", "source_split", "source_id", "sha256", "quarantine_reason"]
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(quarantined)

    summary = {
        split: {name: sum(row["class_name"] == name for row in rows) for name in CLASSES}
        for split, rows in split_records.items()
    }
    summary["invalid_image_count"] = len(invalid)
    summary["quarantined_image_count"] = len(quarantined)
    summary["quarantined_source_family_count"] = len({row["source_id"] for row in quarantined})
    summary["usable_image_count"] = len(usable)
    (output_dir / "data_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))


class FabricDataset(Dataset):
    def __init__(self, manifest_path: Path, transform=None):
        import csv

        with manifest_path.open(newline="", encoding="utf-8-sig") as handle:
            self.records = list(csv.DictReader(handle))
        if not self.records:
            raise ValueError(f"Manifest is empty: {manifest_path}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        row = self.records[index]
        with Image.open(row["path"]) as image:
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image, int(row["label"])


def build_transforms(image_size: int = 224):
    train_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.12, contrast=0.12, saturation=0.10, hue=0.02),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    evaluation_transform = transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )
    return train_transform, evaluation_transform


def main():
    parser = argparse.ArgumentParser(description="Inspect TextileClass-7 and create deterministic manifests.")
    parser.add_argument("--data", type=Path, default=Path("dataset"))
    parser.add_argument("--output", type=Path, default=Path("manifests"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    root = resolve_data_root(args.data)
    prepare_manifests(root, args.output, args.seed)


if __name__ == "__main__":
    main()

