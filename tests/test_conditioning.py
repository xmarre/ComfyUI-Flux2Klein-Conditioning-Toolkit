from __future__ import annotations

import torch

from flux2_klein_conditioning.nodes import (
    Flux2KleinConditioningEnhancer,
    Flux2KleinSectionedTextEncoder,
    Flux2KleinTokenRegionController,
)


class FakeClip:
    def tokenize(self, text: str):
        token_count = min(16, max(2, len(text.split()) + 1))
        mask = torch.zeros(1, 16, dtype=torch.int64)
        mask[:, :token_count] = 1
        return {"input_ids": torch.arange(16).unsqueeze(0), "attention_mask": mask}

    def encode_from_tokens(self, tokens, return_pooled=True):
        mask = tokens["attention_mask"]
        cond = torch.randn(1, mask.shape[1], 8)
        pooled = torch.randn(1, 8)
        return cond, pooled


def test_sectioned_encoder_preserves_attention_mask():
    node = Flux2KleinSectionedTextEncoder()
    conditioning, *_ = node.encode_sectioned(
        FakeClip(),
        mode="manual",
        front_text="cat",
        mid_text="wearing a jacket",
        end_text="studio photo",
        show_preview=False,
    )
    assert "attention_mask" in conditioning[0][1]
    assert int(conditioning[0][1]["attention_mask"].sum().item()) >= 2


def test_dampen_does_not_compound_across_conditioning_items():
    node = Flux2KleinConditioningEnhancer()
    cond = torch.ones(1, 8, 4)
    meta = {"attention_mask": torch.tensor([[1, 1, 1, 1, 0, 0, 0, 0]])}
    conditioning = [(cond.clone(), meta), (cond.clone(), meta)]
    out, = node.enhance(
        conditioning,
        magnitude=2.0,
        preserve_original=0.5,
        preserve_mode="dampen",
        skip_bos=False,
    )
    assert torch.allclose(out[0][0], out[1][0])


def test_region_controller_dampen_does_not_compound():
    node = Flux2KleinTokenRegionController()
    cond = torch.ones(1, 8, 4)
    meta = {"attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])}
    conditioning = [(cond.clone(), meta), (cond.clone(), meta)]
    out, = node.control(
        conditioning,
        front_mult=2.0,
        preserve_original=0.5,
        preserve_mode="dampen",
        skip_bos=False,
    )
    assert torch.allclose(out[0][0], out[1][0])


def test_preserve_original_reduces_prompt_pull_in_edit_mode_even_with_neutral_knobs():
    node = Flux2KleinConditioningEnhancer()
    cond = torch.arange(1, 1 + 1 * 8 * 4, dtype=torch.float32).reshape(1, 8, 4)
    ref = torch.ones(1, 4, 2, 2)
    meta = {"attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]]), "reference_latents": [ref]}
    out, = node.enhance(
        [(cond.clone(), meta)],
        preserve_original=0.5,
        preserve_mode="blend_after",
        skip_bos=False,
    )
    active = out[0][0][:, :6, :]
    expected = cond[:, :6, :] * 0.75
    assert torch.allclose(active, expected)
    assert torch.allclose(out[0][0][:, 6:, :], cond[:, 6:, :])


def test_contrast_preserves_average_token_norm_when_magnitude_is_neutral():
    node = Flux2KleinConditioningEnhancer()
    torch.manual_seed(0)
    cond = torch.randn(1, 8, 16)
    meta = {"attention_mask": torch.tensor([[1, 1, 1, 1, 1, 1, 0, 0]])}
    out, = node.enhance(
        [(cond.clone(), meta)],
        contrast=1.0,
        skip_bos=False,
    )
    before = cond[:, :6, :].norm(dim=-1).mean()
    after = out[0][0][:, :6, :].norm(dim=-1).mean()
    assert torch.isclose(before, after, atol=1e-5, rtol=1e-5)
