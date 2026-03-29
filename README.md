# ComfyUI-Flux2Klein-Conditioning-Toolkit

A clean-room-style rebuild of the useful ideas from `capitan01R/ComfyUI-Flux2Klein-Enhancer`, with the brittle parts removed or replaced.

This repo is not trying to resemble the source repo. It keeps the parts that are technically defensible, rewrites the parts that are structurally weak, and drops the parts whose claimed behavior is not well-supported.

## Why this exists

The source repo contributes two genuinely useful ideas:

1. FLUX.2 Klein text conditioning can be manipulated in a targeted way by editing only the active token region.
2. Reference latents are separate conditioning metadata, not part of the text-conditioning tensor, so they deserve their own control surface.

Those ideas are worth keeping. The implementation around them is where the cleanup was needed.

## Scope

This repo provides six nodes:

- **FLUX.2 Klein Conditioning Enhancer**
  - magnitude / contrast / normalization on the active text-token region
  - optional edit-mode text gain
  - optional preservation blend/dampen logic
- **FLUX.2 Klein Token Region Controller**
  - front / mid / end token-region gains
  - optional custom emphasis window
  - optional preservation blend/dampen logic
- **FLUX.2 Klein Reference Latent Mixer**
  - reliable *weakening / fading / replacement* of reference latents
  - supports **all** reference latents, not just the first one
  - channel masking and spatial fade support
- **FLUX.2 Klein Prompt/Reference Balance**
  - single-slider convenience wrapper
  - balances text gain against reference weakening
- **FLUX.2 Klein Reference Appearance Balancer**
  - separates coarse appearance from fine detail in the reference latent
  - lets you boost low-frequency appearance retention without blindly scaling the full latent
- **FLUX.2 Klein Sectioned Text Encoder**
  - manual or auto-balanced front/mid/end prompt authoring
  - preserves `attention_mask` into conditioning metadata when available

## Key design decisions

### 1. The sectioned encoder keeps the attention mask

The source sectioned encoder returned fresh conditioning with only `pooled_output`, while the source enhancer/detail nodes depend on `attention_mask` to find the real active token range. When that mask is missing, they fall back to a heuristic region length. This rebuild preserves the mask whenever the tokenizer exposes it, so sectioned encoding and downstream token-region control agree with each other.

### 2. No per-item parameter mutation inside the conditioning loop

In the source repo, the `dampen` path mutates node parameters inside the loop over conditioning entries. With multiple conditioning items, later items can get re-dampened using already-modified values. This rebuild computes per-item local values and leaves user inputs unchanged.

### 3. Reference control is honest about what it can reliably do

The source repo exposes reference-latent “strength” as raw multiplicative scaling and also includes a balance node that scales reference latents directly. There is an open issue explicitly questioning whether latent multiply or the repo’s implementation actually works as intended. This rebuild therefore does **not** claim reliable “stronger than baseline” reference locking from raw latent multiplication.

Instead, it provides operations that reliably change the latent *content* rather than only its amplitude:

- mix toward **zeros**
- mix toward **gaussian noise**
- mix toward **per-channel mean**
- optionally only in selected channels / spatial regions

That gives you a dependable weakening/fading control surface without overstating what raw pre-model latent scaling can do.


### 3.5. Washed-out edit colors need reference-stream control, not just text contrast

FLUX.2 Klein image-edit runs often preserve layout more reliably than coarse appearance. When that happens, pushing harder on text conditioning can make the edit more obedient while still letting source chroma wash out.

The root problem is that a raw reference latent is doing multiple jobs at once:

- coarse appearance / illumination / large color fields
- local structural lock / edge detail

If you only have whole-latent weakening or text-side gain, loosening the edit path tends to throw away both together. This rebuild now includes a reference-appearance balancer and a `lowpass_reference` mixer mode so you can relax detail lock while keeping more of the coarse appearance signal.

### 4. Multi-reference support is first-class

FLUX.2 Klein officially supports both single-reference and multi-reference editing. The source repo only edits `reference_latents[0]`. This rebuild updates all references by default and lets you target an individual reference when needed.

### 5. No forced `gc.collect()` / `torch.cuda.empty_cache()` on every call

These nodes work on relatively small conditioning tensors and metadata latents. Forcing global collection on every node invocation is unnecessary overhead and makes the implementation harder to reason about. This rebuild stays on the tensors’ existing device and lets normal PyTorch lifetime rules do the cleanup.

## What was preserved / rewritten / dropped

### Preserved

- active-region text-conditioning enhancement
- front/mid/end token-region control
- sectioned prompt authoring workflow
- the insight that reference latents are a separate control surface

### Rewritten

- all node internals
- active-token detection helpers
- sectioned encoder metadata handling
- reference-latent manipulation semantics
- balance-node semantics

### Fixed

- missing `attention_mask` propagation from sectioned encoding
- repeated dampening across multi-item conditioning lists
- first-reference-only handling
- duplicated / inconsistent negative contrast behavior
- needless device-selection / forced-cache-management complexity

### Dropped

- raw “reference strength > 1 means stronger structure lock” claim
- duplicate standalone text-enhancer node
- legacy naming / backward-compat aliases whose only purpose was historical carry-over
- wide extrapolation-style preserve ranges that blur interpolation semantics

## Installation

```bash
cd ComfyUI/custom_nodes
git clone <this-repo-url> ComfyUI-Flux2Klein-Conditioning-Toolkit
```

Restart ComfyUI.

## Node notes

### FLUX.2 Klein Conditioning Enhancer

Use this when you want broad prompt-following adjustment on the active token region.

Parameters:

- `magnitude`: overall gain on active user tokens
- `contrast`: sharpens or softens token-to-token differences while preserving average token energy
- `normalize_strength`: equalizes token norms
- `preserve_original`: blends some original conditioning back in, and in edit mode also reduces prompt pull against the reference path
- `preserve_mode`:
  - `blend_after`: apply edits, then interpolate with original
  - `dampen`: reduce edit strength before applying edits
  - `hybrid`: dampen first, then blend back
- `edit_text_weight`: extra gain when reference latents are present
- `active_end_override`: manual override when upstream metadata is incomplete
- `skip_bos`: default `True`; BOS should usually not dominate conditioning edits

### FLUX.2 Klein Token Region Controller

Use this when you want rough semantic steering by token position.

It still uses positional regions, not semantic parsing, so the sectioned encoder is provided to make those positions more intentional.

### FLUX.2 Klein Reference Latent Mixer

This node is intentionally framed as a **mixer**, not a “strength booster.”

Parameters:

- `reference_keep`: how much original reference latent to keep
- `replace_mode`:
  - `zeros`: strongest weakening / removal
  - `gaussian_noise`: loosens structure while keeping a noisy latent-like signal
  - `channel_mean`: removes local spatial structure while keeping coarse channel statistics
  - `lowpass_reference`: replaces removed content with a blurred copy of the original reference, which is often better for retaining broad color/illumination cues during edits
- `channel_mask_start` / `channel_mask_end`: restrict effect to latent channels
- `spatial_fade` / `spatial_fade_strength`: fade only part of the spatial field
- `target_reference_index`: `-1` = all references

### FLUX.2 Klein Reference Appearance Balancer

Use this when Klein keeps structure but washes out the source image's broad color/appearance cues during an edit.

Parameters:

- `appearance_scale`: scales the blurred / low-frequency part of the reference latent
- `detail_scale`: scales the residual / high-frequency part of the reference latent
- `blur_radius`: controls how aggressively the latent is split into coarse appearance vs detail
- `channel_mask_start` / `channel_mask_end`: restrict effect to selected latent channels
- `target_reference_index`: `-1` = all references

A practical starting point for washed-out edits is:

- `appearance_scale = 1.15 to 1.35`
- `detail_scale = 0.85 to 1.00`
- `blur_radius = 2 to 4`

This is intentionally more targeted than just increasing text contrast or globally scaling the whole reference latent.

### FLUX.2 Klein Prompt/Reference Balance

This is a convenience node.

- `balance = 0.0`: mute prompt text, keep full reference
- `balance = 0.5`: keep both
- `balance = 1.0`: keep full prompt text, remove reference

Unlike the source repo, the reference side is implemented via reference mixing, not raw latent scaling.

### FLUX.2 Klein Sectioned Text Encoder

- `manual`: use explicit front/mid/end inputs or `[FRONT]...[MID]...[END]` markers
- `auto_balanced`: split the prompt into rough 25/50/25 semantic sections by sentence/comma groups

When the tokenizer exposes an attention mask, this node stores it in conditioning metadata so downstream nodes can operate on the real active token range.

## Tests

The repo includes small pytest tests for the specific regressions this rebuild is targeting:

- sectioned encoder preserves `attention_mask`
- dampen logic does not compound across multiple conditioning entries
- reference mixer updates all references by default
- balance node actually removes reference content in text-only mode

Run them with:

```bash
cd ComfyUI-Flux2Klein-Conditioning-Toolkit
pytest
```

## Assumptions / deviations

- This rebuild assumes ComfyUI’s FLUX.2 Klein conditioning metadata uses the same `reference_latents` key described by the source repo.
- This rebuild assumes tokenizer outputs may expose `attention_mask` in varying container shapes, so extraction is intentionally permissive.
- This rebuild intentionally tightens several parameter ranges to interpolation-style semantics instead of preserving extrapolative values that are harder to reason about.
- This rebuild does **not** attempt model-patch hooks for post-encoder or post-`img_in` reference-stream manipulation. That could be added later, but it is outside the smallest correct scope for this repo.

## Credit

Technical inspiration came from:

- `capitan01R/ComfyUI-Flux2Klein-Enhancer`
- the discussion in source repo issue `#6`

This repo reuses the general ideas, not the structure or the problematic semantics.
