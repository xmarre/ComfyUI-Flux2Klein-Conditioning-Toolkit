from __future__ import annotations

import math

import torch


def apply_contrast(active: torch.Tensor, contrast: float) -> torch.Tensor:
    if contrast == 0.0 or active.numel() == 0:
        return active
    seq_mean = active.mean(dim=1, keepdim=True)
    deviation = active - seq_mean
    scale = 1.0 + contrast if contrast >= 0.0 else math.exp(contrast)
    return seq_mean + deviation * scale


def apply_normalize(active: torch.Tensor, normalize_strength: float, eps: float = 1e-8) -> torch.Tensor:
    strength = float(max(0.0, min(1.0, normalize_strength)))
    if strength == 0.0 or active.numel() == 0:
        return active
    token_norms = active.norm(dim=-1, keepdim=True)
    mean_norm = token_norms.mean()
    normalized = active / (token_norms + eps) * mean_norm
    return active * (1.0 - strength) + normalized * strength


def scale_tensor(active: torch.Tensor, magnitude: float) -> torch.Tensor:
    if magnitude == 1.0:
        return active
    return active * magnitude
