"""ResNet18 transfer-learning model factory."""

from __future__ import annotations

import torch.nn as nn
from torchvision.models import ResNet18_Weights, resnet18


def build_model(num_classes: int = 7, pretrained: bool = True, freeze_backbone: bool = False) -> nn.Module:
    weights = ResNet18_Weights.DEFAULT if pretrained else None
    model = resnet18(weights=weights)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)
    if freeze_backbone:
        for name, parameter in model.named_parameters():
            parameter.requires_grad = name.startswith("fc.")
    return model

