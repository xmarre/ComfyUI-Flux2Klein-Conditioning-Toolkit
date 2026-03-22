from __future__ import annotations

import re
from typing import Any

import torch

from .common import (
    active_end_from_attention_mask,
    apply_preserve_blend,
    clone_meta,
    dampen_toward_neutral,
    detect_active_end,
    detect_active_slice,
    find_attention_mask,
    get_reference_latents,
    set_reference_latents,
)
from .ops import apply_contrast, apply_normalize, scale_tensor
from .reference import mix_reference_latent


CATEGORY = "conditioning/flux2klein"


class Flux2KleinConditioningEnhancer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "magnitude": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
            },
            "optional": {
                "contrast": ("FLOAT", {"default": 0.0, "min": -2.0, "max": 3.0, "step": 0.05}),
                "normalize_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "preserve_original": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "preserve_mode": (["blend_after", "dampen", "hybrid"], {"default": "blend_after"}),
                "edit_text_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "active_end_override": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "skip_bos": ("BOOLEAN", {"default": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "enhance"
    CATEGORY = CATEGORY

    def enhance(
        self,
        conditioning,
        magnitude=1.0,
        contrast=0.0,
        normalize_strength=0.0,
        preserve_original=0.0,
        preserve_mode="blend_after",
        edit_text_weight=1.0,
        active_end_override=0,
        skip_bos=True,
        debug=False,
    ):
        if not conditioning:
            return (conditioning,)

        no_op = (
            magnitude == 1.0
            and contrast == 0.0
            and normalize_strength == 0.0
            and preserve_original == 0.0
            and edit_text_weight == 1.0
        )
        if no_op:
            return (conditioning,)

        output = []
        for idx, (cond_tensor, meta) in enumerate(conditioning):
            if not isinstance(cond_tensor, torch.Tensor) or cond_tensor.ndim != 3:
                output.append((cond_tensor, meta))
                continue

            cond = cond_tensor.to(dtype=torch.float32).clone()
            start, end = detect_active_slice(meta, cond.shape[1], skip_bos=skip_bos, override=active_end_override)
            if start >= end:
                output.append((cond_tensor, meta))
                continue

            is_edit_mode = bool(get_reference_latents(meta))
            original_active = cond[:, start:end, :].clone()
            active = original_active.clone()

            local_magnitude = float(magnitude)
            local_contrast = float(contrast)
            local_normalize = float(normalize_strength)
            local_edit_weight = float(edit_text_weight)

            if preserve_mode in {"dampen", "hybrid"} and preserve_original > 0.0:
                local_magnitude = dampen_toward_neutral(local_magnitude, 1.0, preserve_original)
                local_contrast = dampen_toward_neutral(local_contrast, 0.0, preserve_original)
                local_normalize = dampen_toward_neutral(local_normalize, 0.0, preserve_original)
                local_edit_weight = dampen_toward_neutral(local_edit_weight, 1.0, preserve_original)

            active = apply_contrast(active, local_contrast)
            active = apply_normalize(active, local_normalize)
            active = scale_tensor(active, local_magnitude)
            if is_edit_mode:
                active = scale_tensor(active, local_edit_weight)

            if preserve_mode in {"blend_after", "hybrid"} and preserve_original > 0.0:
                active = apply_preserve_blend(active, original_active, preserve_original)

            cond[:, start:end, :] = active
            if debug:
                print(
                    f"[Flux2KleinConditioningEnhancer] item={idx} active=[{start}:{end}] "
                    f"edit_mode={is_edit_mode} mean_delta={(active - original_active).abs().mean().item():.6f}"
                )
            output.append((cond.to(dtype=cond_tensor.dtype), meta))
        return (output,)


class Flux2KleinTokenRegionController:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
            },
            "optional": {
                "front_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "mid_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "end_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "emphasis_start": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "emphasis_end": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "emphasis_mult": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 3.0, "step": 0.05}),
                "preserve_original": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05}),
                "preserve_mode": (["blend_after", "dampen", "hybrid"], {"default": "blend_after"}),
                "active_end_override": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "skip_bos": ("BOOLEAN", {"default": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "control"
    CATEGORY = CATEGORY

    def control(
        self,
        conditioning,
        front_mult=1.0,
        mid_mult=1.0,
        end_mult=1.0,
        emphasis_start=0,
        emphasis_end=0,
        emphasis_mult=1.0,
        preserve_original=0.0,
        preserve_mode="blend_after",
        active_end_override=0,
        skip_bos=True,
        debug=False,
    ):
        if not conditioning:
            return (conditioning,)

        no_op = (
            front_mult == 1.0
            and mid_mult == 1.0
            and end_mult == 1.0
            and (emphasis_end == 0 or emphasis_mult == 1.0)
            and preserve_original == 0.0
        )
        if no_op:
            return (conditioning,)

        output = []
        for idx, (cond_tensor, meta) in enumerate(conditioning):
            if not isinstance(cond_tensor, torch.Tensor) or cond_tensor.ndim != 3:
                output.append((cond_tensor, meta))
                continue

            cond = cond_tensor.to(dtype=torch.float32).clone()
            start, end = detect_active_slice(meta, cond.shape[1], skip_bos=skip_bos, override=active_end_override)
            if start >= end:
                output.append((cond_tensor, meta))
                continue

            original_active = cond[:, start:end, :].clone()
            active = original_active.clone()
            num_tokens = active.shape[1]

            local_front = float(front_mult)
            local_mid = float(mid_mult)
            local_end = float(end_mult)
            local_emphasis = float(emphasis_mult)
            if preserve_mode in {"dampen", "hybrid"} and preserve_original > 0.0:
                local_front = dampen_toward_neutral(local_front, 1.0, preserve_original)
                local_mid = dampen_toward_neutral(local_mid, 1.0, preserve_original)
                local_end = dampen_toward_neutral(local_end, 1.0, preserve_original)
                local_emphasis = dampen_toward_neutral(local_emphasis, 1.0, preserve_original)

            front_end = int(num_tokens * 0.25)
            mid_end = int(num_tokens * 0.75)

            if local_front != 1.0 and front_end > 0:
                active[:, :front_end, :] *= local_front
            if local_mid != 1.0 and mid_end > front_end:
                active[:, front_end:mid_end, :] *= local_mid
            if local_end != 1.0 and num_tokens > mid_end:
                active[:, mid_end:, :] *= local_end

            if emphasis_end > 0 and local_emphasis != 1.0:
                emp_start = max(0, min(int(emphasis_start), num_tokens - 1))
                emp_end = max(emp_start + 1, min(int(emphasis_end), num_tokens))
                active[:, emp_start:emp_end, :] *= local_emphasis
            else:
                emp_start = emp_end = 0

            if preserve_mode in {"blend_after", "hybrid"} and preserve_original > 0.0:
                active = apply_preserve_blend(active, original_active, preserve_original)

            cond[:, start:end, :] = active
            if debug:
                print(
                    f"[Flux2KleinTokenRegionController] item={idx} active=[{start}:{end}] regions=(0:{front_end},{front_end}:{mid_end},{mid_end}:{num_tokens}) "
                    f"emphasis=[{emp_start}:{emp_end}] mean_delta={(active - original_active).abs().mean().item():.6f}"
                )
            output.append((cond.to(dtype=cond_tensor.dtype), meta))
        return (output,)


class Flux2KleinReferenceLatentMixer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "reference_keep": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "replace_mode": (["zeros", "gaussian_noise", "channel_mean"], {"default": "zeros"}),
                "channel_mask_start": ("INT", {"default": 0, "min": 0, "max": 4096, "step": 1}),
                "channel_mask_end": ("INT", {"default": 128, "min": 0, "max": 4096, "step": 1}),
                "spatial_fade": (["none", "center_out", "edges_out", "top_down", "left_right"], {"default": "none"}),
                "spatial_fade_strength": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
                "target_reference_index": ("INT", {"default": -1, "min": -1, "max": 64, "step": 1}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "control"
    CATEGORY = CATEGORY

    def control(
        self,
        conditioning,
        reference_keep=1.0,
        replace_mode="zeros",
        channel_mask_start=0,
        channel_mask_end=128,
        spatial_fade="none",
        spatial_fade_strength=0.5,
        target_reference_index=-1,
        debug=False,
    ):
        if not conditioning:
            return (conditioning,)

        if reference_keep == 1.0 and spatial_fade == "none":
            return (conditioning,)

        output = []
        for idx, (cond_tensor, meta) in enumerate(conditioning):
            refs = get_reference_latents(meta)
            if not refs:
                output.append((cond_tensor, meta))
                continue

            new_refs = []
            for ref_idx, ref in enumerate(refs):
                if target_reference_index >= 0 and ref_idx != target_reference_index:
                    new_refs.append(ref.clone())
                    continue
                mixed = mix_reference_latent(
                    ref,
                    reference_keep=float(reference_keep),
                    replace_mode=replace_mode,
                    channel_start=int(channel_mask_start),
                    channel_end=int(channel_mask_end),
                    spatial_fade=spatial_fade,
                    spatial_fade_strength=float(spatial_fade_strength),
                )
                if debug:
                    delta = (mixed.to(torch.float32) - ref.to(torch.float32)).abs().mean().item()
                    print(
                        f"[Flux2KleinReferenceLatentMixer] item={idx} ref={ref_idx} keep={reference_keep:.2f} "
                        f"mode={replace_mode} mean_delta={delta:.6f}"
                    )
                new_refs.append(mixed)
            output.append((cond_tensor, set_reference_latents(meta, new_refs)))
        return (output,)


class Flux2KleinPromptReferenceBalance:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "conditioning": ("CONDITIONING",),
                "balance": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.05}),
            },
            "optional": {
                "replace_mode": (["zeros", "gaussian_noise", "channel_mean"], {"default": "zeros"}),
                "skip_bos": ("BOOLEAN", {"default": True}),
                "active_end_override": ("INT", {"default": 0, "min": 0, "max": 512, "step": 1}),
                "target_reference_index": ("INT", {"default": -1, "min": -1, "max": 64, "step": 1}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "balance"
    CATEGORY = CATEGORY

    def balance(
        self,
        conditioning,
        balance=0.5,
        replace_mode="zeros",
        skip_bos=True,
        active_end_override=0,
        target_reference_index=-1,
        debug=False,
    ):
        if not conditioning:
            return (conditioning,)

        balance = float(max(0.0, min(1.0, balance)))
        if balance <= 0.5:
            text_gain = balance * 2.0
            reference_keep = 1.0
        else:
            text_gain = 1.0
            reference_keep = (1.0 - balance) * 2.0

        output = []
        for idx, (cond_tensor, meta) in enumerate(conditioning):
            modified_cond = cond_tensor.clone()
            if isinstance(cond_tensor, torch.Tensor) and cond_tensor.ndim == 3 and text_gain != 1.0:
                cond = cond_tensor.to(dtype=torch.float32).clone()
                start, end = detect_active_slice(meta, cond.shape[1], skip_bos=skip_bos, override=active_end_override)
                if start < end:
                    cond[:, start:end, :] *= text_gain
                    modified_cond = cond.to(dtype=cond_tensor.dtype)

            refs = get_reference_latents(meta)
            if refs and reference_keep != 1.0:
                new_refs = []
                for ref_idx, ref in enumerate(refs):
                    if target_reference_index >= 0 and ref_idx != target_reference_index:
                        new_refs.append(ref.clone())
                        continue
                    new_refs.append(
                        mix_reference_latent(
                            ref,
                            reference_keep=reference_keep,
                            replace_mode=replace_mode,
                            channel_start=0,
                            channel_end=ref.shape[1],
                            spatial_fade="none",
                            spatial_fade_strength=0.0,
                        )
                    )
                new_meta = set_reference_latents(meta, new_refs)
            else:
                new_meta = clone_meta(meta)

            if debug:
                print(
                    f"[Flux2KleinPromptReferenceBalance] item={idx} balance={balance:.2f} "
                    f"text_gain={text_gain:.2f} reference_keep={reference_keep:.2f}"
                )
            output.append((modified_cond, new_meta))
        return (output,)


class Flux2KleinSectionedTextEncoder:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip": ("CLIP",),
                "mode": (["manual", "auto_balanced"], {"default": "manual"}),
            },
            "optional": {
                "front_text": ("STRING", {"multiline": True, "default": ""}),
                "mid_text": ("STRING", {"multiline": True, "default": ""}),
                "end_text": ("STRING", {"multiline": True, "default": ""}),
                "combined_prompt": ("STRING", {"multiline": True, "default": ""}),
                "separator": (["comma", "period", "space", "newline"], {"default": "comma"}),
                "show_preview": ("BOOLEAN", {"default": True}),
                "debug": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("CONDITIONING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("conditioning", "front_section", "mid_section", "end_section", "full_prompt")
    FUNCTION = "encode_sectioned"
    CATEGORY = CATEGORY

    _SECTION_PATTERN = re.compile(r"\[(FRONT|MID|END)\](.*?)(?=\[(?:FRONT|MID|END)\]|$)", re.DOTALL | re.IGNORECASE)

    def _parse_manual_sections(self, text: str) -> dict[str, str] | None:
        matches = self._SECTION_PATTERN.findall(text)
        sections = {"front": "", "mid": "", "end": ""}
        for section_name, content in matches:
            sections[section_name.lower()] = content.strip()
        return sections if any(sections.values()) else None

    def _auto_balance_sections(self, text: str) -> dict[str, str]:
        if not text or not text.strip():
            return {"front": "", "mid": "", "end": ""}

        parts = [part.strip() for part in re.split(r"(?<=[.!?])\s+", text) if part.strip()]
        if len(parts) <= 1:
            parts = [part.strip() for part in text.split(",") if part.strip()]
        if not parts:
            return {"front": text.strip(), "mid": "", "end": ""}

        total = len(parts)
        front_count = max(1, int(total * 0.25))
        mid_count = max(1, int(total * 0.5))
        return {
            "front": " ".join(parts[:front_count]).strip(),
            "mid": " ".join(parts[front_count:front_count + mid_count]).strip(),
            "end": " ".join(parts[front_count + mid_count:]).strip(),
        }

    def _join_sections(self, sections: dict[str, str], separator: str) -> str:
        sep_map = {
            "comma": ", ",
            "period": ". ",
            "space": " ",
            "newline": "\n",
        }
        parts = [sections[name] for name in ("front", "mid", "end") if sections[name]]
        return sep_map.get(separator, ", ").join(parts)

    def _section_token_estimate(self, clip: Any, text: str) -> int:
        if not text.strip():
            return 0
        try:
            tokenized = clip.tokenize(text)
            attn_mask = find_attention_mask(tokenized)
            if attn_mask is not None:
                active_end = active_end_from_attention_mask(attn_mask, int(attn_mask.shape[-1]))
                if active_end is not None:
                    return int(active_end)
        except Exception:
            pass
        return max(1, int(len(text) / 3.5))

    def encode_sectioned(
        self,
        clip,
        mode="manual",
        front_text="",
        mid_text="",
        end_text="",
        combined_prompt="",
        separator="comma",
        show_preview=True,
        debug=False,
    ):
        if mode == "manual":
            parsed = self._parse_manual_sections(combined_prompt) if combined_prompt else None
            sections = parsed or {"front": front_text, "mid": mid_text, "end": end_text}
        else:
            text_to_balance = combined_prompt if combined_prompt else f"{front_text} {mid_text} {end_text}".strip()
            sections = self._auto_balance_sections(text_to_balance)

        final_prompt = self._join_sections(sections, separator)
        tokenized = clip.tokenize(final_prompt)
        cond, pooled = clip.encode_from_tokens(tokenized, return_pooled=True)
        meta = {"pooled_output": pooled}
        attn_mask = find_attention_mask(tokenized)
        if attn_mask is not None:
            meta["attention_mask"] = attn_mask.clone() if isinstance(attn_mask, torch.Tensor) else attn_mask
        conditioning = [[cond, meta]]

        if show_preview or debug:
            actual_active_end = detect_active_end(meta, cond.shape[1], fallback=cond.shape[1])
            front_est = self._section_token_estimate(clip, sections["front"])
            mid_est = self._section_token_estimate(clip, sections["mid"])
            end_est = self._section_token_estimate(clip, sections["end"])
            print("\n" + "=" * 72)
            print("FLUX.2 KLEIN SECTIONED TEXT ENCODER")
            print("=" * 72)
            print(f"mode={mode} separator={separator} actual_active_tokens={actual_active_end}")
            print(f"front~{front_est}: {sections['front']!r}")
            print(f"mid~{mid_est}: {sections['mid']!r}")
            print(f"end~{end_est}: {sections['end']!r}")
            print("-" * 72)
            print(final_prompt)
            print("=" * 72 + "\n")

        return (conditioning, sections["front"], sections["mid"], sections["end"], final_prompt)


NODE_CLASS_MAPPINGS = {
    "Flux2KleinConditioningEnhancer": Flux2KleinConditioningEnhancer,
    "Flux2KleinTokenRegionController": Flux2KleinTokenRegionController,
    "Flux2KleinReferenceLatentMixer": Flux2KleinReferenceLatentMixer,
    "Flux2KleinPromptReferenceBalance": Flux2KleinPromptReferenceBalance,
    "Flux2KleinSectionedTextEncoder": Flux2KleinSectionedTextEncoder,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Flux2KleinConditioningEnhancer": "FLUX.2 Klein Conditioning Enhancer",
    "Flux2KleinTokenRegionController": "FLUX.2 Klein Token Region Controller",
    "Flux2KleinReferenceLatentMixer": "FLUX.2 Klein Reference Latent Mixer",
    "Flux2KleinPromptReferenceBalance": "FLUX.2 Klein Prompt/Reference Balance",
    "Flux2KleinSectionedTextEncoder": "FLUX.2 Klein Sectioned Text Encoder",
}
