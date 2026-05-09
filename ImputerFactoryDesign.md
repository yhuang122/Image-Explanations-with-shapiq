# ImageImputer — Module Implementation Status

> Last updated: 2026-05-09

## Implementation Progress Summary

| Module | Component | Status | Notes |
|---|---|---|---|
| **Data Types** | `SpatialLayout` | ✅ Done | Player↔pixel/token mapping metadata |
| | `PhysicalMask` | ✅ Done | Concrete masks: `image_binary_mask` (N,C,H,W) + `text_attention_mask` (N,L) |
| | `ProcessorOutput` | ✅ Done | Standardized HuggingFace inputs wrapper |
| **Segmenters** | `BaseSegmenter` | ✅ Done | Abstract: `get_layout()` + `generate_masks()` |
| | `PatchSegmenter` | ✅ Done | Rigid grid, supports CLIP/SigLIP/SigLIP2 text masking |
| | `SLICSegmenter` | ⏳ Stub | CPU blueprint only, needs GPU execution path |
| | `GradientGuidedSegmenter` | ⏳ Stub | Needs gradient extraction + watershed layout |
| | `AdaptiveSegmenter` | ⏳ Stub | Needs coarse-to-fine subdivision logic |
| | `HybridSegmenter` | ⏳ Stub | Composition of Gradient + Adaptive |
| **Maskers** | `BaseMasker` | ✅ Done | Abstract: `apply(ProcessorOutput, PhysicalMask)` |
| | `CrossModalMeanMasker` | ✅ Done | Multiplicative binary mask (image) + attention_mask swap (text) |
| | `AttentionMasker` | ⏳ Stub | Needs negative-infinity self-attention injection |
| **Core** | `ImageImputer` | ✅ Done | `forward_1d` + `forward_crossmodal` with batching & device mgmt |
| **Factory** | `ImageImputerFactory` | ✅ Done | Auto-detect model type, assemble PatchSegmenter + CrossModalMeanMasker |
| **Adapters** | `TensorOps` / `TorchOps` / `JaxOps` | ⏳ Stub | Interface defined, implementations pending |
| **Integration** | `VisionLanguageGame` | ✅ Done | Thin adapter: delegates to Imputer, ~75 lines |

### Legend
- ✅ Done — fully implemented and tested
- ⏳ Stub — skeleton exists, logic outstanding
- ❌ Not started — not yet created

---

## Data Transfer Contract

```
Coalitions (np.bool)                Visualization / Notebook
        │                                    │
        ▼                                    ▼
┌─────────────────┐              ┌─────────────────────┐
│   Segmenter     │              │  VisionLanguageGame  │
│  get_layout()   │──────────────│   (thin adapter)     │
│  generate_masks │              │  inputs / processor  │
└────────┬────────┘              │  value_function()    │
         │                       └──────────┬──────────┘
         ▼                                  │
   PhysicalMask                             │
         │                                  │
         ▼                                  ▼
┌─────────────────┐              ┌─────────────────────┐
│    Masker       │              │   ImageImputer      │
│  apply()        │◄─────────────│  forward_1d()       │
└────────┬────────┘              │  forward_crossmodal()│
         │                       └─────────────────────┘
         ▼
   ProcessorOutput (modified) ───► model.forward() ───► np.array
```

### Key Design Decisions

1. **"CPU Planning, GPU Execution"**: Segmenters produce integer index maps once on CPU (via skimage). Thousands of coalition→mask translations happen purely on GPU via native tensor ops.

2. **Imputer owns the inputs**: `ImageImputer` stores `inputs_original` (ProcessorOutput), `inputs_raw` (HF dict for `.tokens()`), and `input_image`/`input_text` (for crossmodal edge cases where batch sizes diverge).

3. **Game is a thin shell**: `VisionLanguageGame` delegates all masking/batching/model-forward to the Imputer. It only handles shapiq scheduling (normalization values, player counts).

---

## Current Implementation Details

### `ImputerFactory/data.py`
Three dataclasses serve as the universal data protocol:
- **`SpatialLayout`**: Immutable metadata describing the spatial division. Produced once by Segmenter, consumed by Imputer.
- **`PhysicalMask`**: Concrete tensor masks. `image_binary_mask` (N, C, H, W) float + `text_attention_mask` (N, L) int.
- **`ProcessorOutput`**: Wraps `pixel_values`, `input_ids`, `attention_mask` with a `to_dict()` for model forwarding.

### `ImputerFactory/segmenters/patch.py` — PatchSegmenter
- Pre-computes `SpatialLayout` at init (is_stateful=False)
- `generate_masks()` converts coalition arrays → `PhysicalMask`
- Image: expand patch-level booleans → `patch_size×patch_size` blocks → (N, C, H, W)
- Text: handles CLIP (BOS/EOS wrapping) vs SigLIP (right-padding) mask formats

### `ImputerFactory/maskers/mean.py` — CrossModalMeanMasker
- Image: `pixel_values *= image_binary_mask` (zero-mean normalization → mean fill)
- Text: replaces `attention_mask` with coalition-derived mask
- Clones inputs to avoid mutation

### `ImputerFactory/core/imputer.py` — ImageImputer
- **`forward_1d(coalitions, batch_size)`**: Splits coalitions → generates masks → batches → masks → model → extracts diagonal
- **`forward_crossmodal(coalitions_img, coalitions_txt, batch_size)`**: Double loop (image outer, text inner). Edge case: when txt_bs ≠ img_bs, re-processes via `_preprocess_batch()` using stored `input_image`/`input_text`
- **`_model_forward()`**: Auto-detects model device, moves inputs before forward
- Stores: `inputs_original`, `inputs_raw`, `input_image`, `input_text`, `model`, `processor`, `segmenter`, `masker`, `layout`

### `ImputerFactory/factory.py` — ImageImputerFactory
- `build(model, processor, input_image, input_text, accelerator=None)`:
  1. Infers model type (clip/siglip/siglip2)
  2. Preprocesses once to determine `n_players_text` + `text_total_length`
  3. Creates `PatchSegmenter` (baseline) or raises `NotImplementedError` for accelerators
  4. Creates `CrossModalMeanMasker`
  5. Wires `ProcessorOutput` + raw dict + raw image/text into `ImageImputer`

### `Game/game_huggingface.py` — VisionLanguageGame
- Constructor: `VisionLanguageGame(imputer, batch_size=64, verbose=False)`
- `n_players_image` / `n_players_text` from imputer layout
- `inputs` / `processor` properties delegate to imputer (backward compat)
- `value_function()` → `imputer.forward_1d()`
- `value_function_crossmodal()` → `imputer.forward_crossmodal()`

---

## Future Work

### High Priority
- [ ] **Numerical equivalence test**: Compare imputer output vs original built-in path on the same coalitions (tolerance < 1e-6)
- [ ] **Migrate remaining 7 experiment files** to new API (see `experiments/*.py` — all use old `VisionLanguageGame(model, processor, ...)` signature)

### Medium Priority — Accelerators
- [ ] **`GradientGuidedSegmenter`**: Extract gradient map from model → skimage watershed → non-uniform static layout
- [ ] **`AdaptiveSegmenter`**: Coarse grid → score-driven subdivision → feedback loop (requires Imputer ↔ Segmenter state protocol for `is_stateful=True`)
- [ ] **`HybridSegmenter`**: Composition of GradientGuided (initial) + Adaptive (refine)

### Medium Priority — Maskers
- [ ] **`AttentionMasker`**: Intercept self-attention with negative-infinity mask matrices. Requires PyTorch hook injection or HuggingFace `output_attentions` + `attention_mask` override

### Low Priority — Backends
- [ ] **`JaxOps`**: JAX-native tensor ops for Google TPU / JAX model support
- [ ] **`TorchOps`**: Extract PyTorch-specific ops (currently inline) into adapter

### Low Priority — Model Support
- [ ] **CNN models**: `SLICSegmenter` + `CrossModalMeanMasker` baseline
- [ ] **OpenAI API models**: `game_openai.py` integration

### Known Issues
- `forward_crossmodal` reprocesses inputs via HF processor on edge case batches (slight overhead vs original which also did this)
- `_repeat_inputs` uses `.expand().clone()` which duplicates memory; could be optimized with stride tricks
- No mixed-precision (AMP) support yet — relevant for larger models