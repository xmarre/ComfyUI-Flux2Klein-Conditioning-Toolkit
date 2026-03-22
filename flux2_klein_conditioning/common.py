from __future__ import annotations

from typing import Any, Iterable, Optional

import torch


def clone_meta(meta: dict[str, Any]) -> dict[str, Any]:
    cloned = dict(meta)
    refs = meta.get("reference_latents")
    if isinstance(refs, (list, tuple)):
        cloned["reference_latents"] = [ref.clone() if isinstance(ref, torch.Tensor) else ref for ref in refs]
    elif isinstance(refs, torch.Tensor):
        cloned["reference_latents"] = refs.clone()
    return cloned


def _mask_to_tensor(mask: Any) -> Optional[torch.Tensor]:
    if mask is None:
        return None
    if isinstance(mask, torch.Tensor):
        return mask
    if isinstance(mask, (list, tuple)):
        try:
            return torch.as_tensor(mask)
        except Exception:
            return None
    return None


def find_attention_mask(tokenized: Any) -> Optional[torch.Tensor]:
    if isinstance(tokenized, dict):
        if "attention_mask" in tokenized:
            return _mask_to_tensor(tokenized["attention_mask"])
        for value in tokenized.values():
            found = find_attention_mask(value)
            if found is not None:
                return found
        return None
    if isinstance(tokenized, (list, tuple)):
        for value in tokenized:
            found = find_attention_mask(value)
            if found is not None:
                return found
    return None


def active_end_from_attention_mask(attn_mask: Any, seq_len: int) -> Optional[int]:
    mask = _mask_to_tensor(attn_mask)
    if mask is None:
        return None
    if mask.ndim == 0:
        return None
    if mask.ndim > 1:
        mask = mask[0]
    mask = mask.reshape(-1)
    if mask.numel() == 0:
        return None
    positives = (mask > 0).nonzero(as_tuple=False)
    if positives.numel() == 0:
        return None
    return min(int(positives[-1].item()) + 1, seq_len)


def detect_active_end(meta: dict[str, Any], seq_len: int, override: int = 0, fallback: int = 77) -> int:
    if override > 0:
        return max(0, min(int(override), seq_len))
    active_end = active_end_from_attention_mask(meta.get("attention_mask"), seq_len)
    if active_end is not None:
        return active_end
    return min(seq_len, fallback)


def detect_active_slice(meta: dict[str, Any], seq_len: int, *, skip_bos: bool = True, override: int = 0, fallback: int = 77) -> tuple[int, int]:
    end = detect_active_end(meta, seq_len, override=override, fallback=fallback)
    start = 1 if skip_bos and end > 1 else 0
    start = min(start, end)
    return start, end


def get_reference_latents(meta: dict[str, Any]) -> list[torch.Tensor]:
    refs = meta.get("reference_latents")
    if refs is None:
        return []
    if isinstance(refs, torch.Tensor):
        return [refs]
    if isinstance(refs, (list, tuple)):
        return [ref for ref in refs if isinstance(ref, torch.Tensor)]
    return []


def set_reference_latents(meta: dict[str, Any], refs: Iterable[torch.Tensor]) -> dict[str, Any]:
    new_meta = clone_meta(meta)
    new_meta["reference_latents"] = list(refs)
    return new_meta


def apply_preserve_blend(enhanced: torch.Tensor, original: torch.Tensor, preserve_original: float) -> torch.Tensor:
    preserve = float(max(0.0, min(1.0, preserve_original)))
    return enhanced * (1.0 - preserve) + original * preserve


def dampen_toward_neutral(value: float, neutral: float, preserve_original: float) -> float:
    preserve = float(max(0.0, min(1.0, preserve_original)))
    return neutral + (value - neutral) * (1.0 - preserve)
