from __future__ import annotations

import torch

from flux2_klein_conditioning.nodes import Flux2KleinPromptReferenceBalance, Flux2KleinReferenceLatentMixer


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
