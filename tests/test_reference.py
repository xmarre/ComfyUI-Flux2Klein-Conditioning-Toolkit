from __future__ import annotations

import torch

from flux2_klein_conditioning.nodes import (
    Flux2KleinPromptReferenceBalance,
    Flux2KleinReferenceAppearanceBalancer,
    Flux2KleinReferenceLatentMixer,
)
from flux2_klein_conditioning.reference import gaussian_blur_per_channel


def test_gaussian_blur_preserves_constant_tensor():
    x = torch.ones(1, 4, 8, 8, dtype=torch.float32)
    y = gaussian_blur_per_channel(x, 2)
    assert torch.allclose(x, y, atol=1e-6, rtol=1e-6)


def test_reference_mixer_updates_all_refs_by_default():
    node = Flux2KleinReferenceLatentMixer()
    ref_a = torch.ones(1, 4, 2, 2)
    ref_b = torch.ones(1, 4, 2, 2) * 2
    conditioning = [(torch.zeros(1, 4, 4), {"reference_latents": [ref_a, ref_b]})]
    out, = node.control(conditioning, reference_keep=0.0, replace_mode="zeros", channel_mask_end=4)
    refs = out[0][1]["reference_latents"]
    assert torch.count_nonzero(refs[0]) == 0
    assert torch.count_nonzero(refs[1]) == 0


def test_reference_mixer_can_target_single_ref():
    node = Flux2KleinReferenceLatentMixer()
    ref_a = torch.ones(1, 4, 2, 2)
    ref_b = torch.ones(1, 4, 2, 2) * 2
    conditioning = [(torch.zeros(1, 4, 4), {"reference_latents": [ref_a, ref_b]})]
    out, = node.control(
        conditioning,
        reference_keep=0.0,
        replace_mode="zeros",
        channel_mask_end=4,
        target_reference_index=1,
    )
    refs = out[0][1]["reference_latents"]
    assert torch.count_nonzero(refs[0]) == ref_a.numel()
    assert torch.count_nonzero(refs[1]) == 0


def test_prompt_reference_balance_reduces_reference_when_balance_is_text_only():
    node = Flux2KleinPromptReferenceBalance()
    cond = torch.ones(1, 8, 4)
    ref = torch.ones(1, 4, 2, 2)
    meta = {"attention_mask": torch.tensor([[1, 1, 1, 1, 1, 0, 0, 0]]), "reference_latents": [ref]}
    out, = node.balance([(cond, meta)], balance=1.0, replace_mode="zeros", skip_bos=False)
    assert torch.count_nonzero(out[0][1]["reference_latents"][0]) == 0


def test_reference_mixer_lowpass_mode_preserves_nonzero_coarse_signal():
    node = Flux2KleinReferenceLatentMixer()
    ref = torch.arange(1, 1 + 1 * 4 * 6 * 6, dtype=torch.float32).reshape(1, 4, 6, 6)
    conditioning = [(torch.zeros(1, 4, 4), {"reference_latents": [ref]})]
    out, = node.control(conditioning, reference_keep=0.0, replace_mode="lowpass_reference", channel_mask_end=4)
    mixed = out[0][1]["reference_latents"][0]
    assert torch.count_nonzero(mixed) > 0
    assert not torch.allclose(mixed, ref)


def test_reference_appearance_balancer_boosts_coarse_component_and_reduces_detail():
    node = Flux2KleinReferenceAppearanceBalancer()
    base_x = torch.linspace(0.0, 1.0, 8).view(1, 1, 1, 8).expand(1, 4, 8, 8)
    base_y = torch.linspace(0.0, 1.0, 8).view(1, 1, 8, 1).expand(1, 4, 8, 8)
    coarse = base_x + base_y
    checker = ((torch.arange(8).view(1, 1, 8, 1) + torch.arange(8).view(1, 1, 1, 8)) % 2).float() * 2.0 - 1.0
    detail = checker.expand_as(coarse) * 0.25
    ref = coarse + detail
    conditioning = [(torch.zeros(1, 4, 4), {"reference_latents": [ref]})]

    out, = node.balance(conditioning, appearance_scale=1.5, detail_scale=0.25, blur_radius=2, channel_mask_end=4)
    modified = out[0][1]["reference_latents"][0]

    input_low = gaussian_blur_per_channel(ref, 2)
    output_low = gaussian_blur_per_channel(modified, 2)
    input_high = ref - input_low
    output_high = modified - output_low

    assert output_low.abs().mean() > input_low.abs().mean()
    assert output_high.abs().mean() < input_high.abs().mean()
