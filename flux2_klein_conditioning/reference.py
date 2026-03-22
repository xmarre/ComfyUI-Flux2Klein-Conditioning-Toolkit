from __future__ import annotations

from typing import Literal

import torch

ReplaceMode = Literal["zeros", "gaussian_noise", "channel_mean"]
FadeMode = Literal["none", "center_out", "edges_out", "top_down", "left_right"]


def create_spatial_mask(h: int, w: int, mode: FadeMode, strength: float, *, device: torch.device) -> torch.Tensor:
    strength = float(max(0.0, min(1.0, strength)))
    if mode == "none":
        return torch.ones((1, 1, h, w), dtype=torch.float32, device=device)

    y = torch.linspace(0.0, 1.0, h, device=device, dtype=torch.float32).unsqueeze(1).expand(h, w)
    x = torch.linspace(0.0, 1.0, w, device=device, dtype=torch.float32).unsqueeze(0).expand(h, w)

    if mode == "center_out":
        dist = torch.sqrt((y - 0.5) ** 2 + (x - 0.5) ** 2)
        dist = dist / max(float(dist.max().item()), 1e-8)
        mask = 1.0 - dist * strength
    elif mode == "edges_out":
        dist = torch.sqrt((y - 0.5) ** 2 + (x - 0.5) ** 2)
        dist = dist / max(float(dist.max().item()), 1e-8)
        mask = (1.0 - strength) + dist * strength
    elif mode == "top_down":
        mask = 1.0 - y * strength
    elif mode == "left_right":
        mask = 1.0 - x * strength
    else:
        mask = torch.ones((h, w), dtype=torch.float32, device=device)

    return mask.clamp(0.0, 1.0).unsqueeze(0).unsqueeze(0)


def build_replacement(selected: torch.Tensor, mode: ReplaceMode) -> torch.Tensor:
    if mode == "zeros":
        return torch.zeros_like(selected)
    if mode == "gaussian_noise":
        scale = selected.std().clamp_min(1e-8)
        return torch.randn_like(selected) * scale
    if mode == "channel_mean":
        return selected.mean(dim=(-2, -1), keepdim=True).expand_as(selected)
    raise ValueError(f"Unsupported replacement mode: {mode}")


def mix_reference_latent(
    ref: torch.Tensor,
    *,
    reference_keep: float,
    replace_mode: ReplaceMode,
    channel_start: int,
    channel_end: int,
    spatial_fade: FadeMode,
    spatial_fade_strength: float,
) -> torch.Tensor:
    if ref.ndim != 4:
        raise ValueError(f"Expected [B, C, H, W] reference latent, got shape {tuple(ref.shape)}")

    keep = float(max(0.0, min(1.0, reference_keep)))
    if keep == 1.0 and spatial_fade == "none":
        return ref

    working = ref.to(dtype=torch.float32)
    _, channels, h, w = working.shape
    ch_start = max(0, min(int(channel_start), channels))
    ch_end = max(ch_start, min(int(channel_end), channels))
    if ch_start == ch_end:
        return working.to(dtype=ref.dtype)

    mask = create_spatial_mask(h, w, spatial_fade, spatial_fade_strength, device=working.device)
    full_keep = torch.ones_like(working)
    full_keep[:, ch_start:ch_end, :, :] = mask.expand(-1, ch_end - ch_start, -1, -1) * keep + (1.0 - mask.expand(-1, ch_end - ch_start, -1, -1))

    selected = working[:, ch_start:ch_end, :, :]
    replacement = build_replacement(selected, replace_mode)
    mixed = selected * full_keep[:, ch_start:ch_end, :, :] + replacement * (1.0 - full_keep[:, ch_start:ch_end, :, :])

    result = working.clone()
    result[:, ch_start:ch_end, :, :] = mixed
    return result.to(dtype=ref.dtype)
