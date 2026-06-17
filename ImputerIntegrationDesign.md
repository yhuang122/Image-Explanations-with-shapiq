# Vision Imputer — Architecture & Module Reference

`shapiq/imputer/vision/` — Pluggable modular pipeline for vision-language model
explanation via Shapley interactions.

---

## 1. Architecture Overview

```
                         ┌──────────────────┐
                         │VisionImputerFactory│  ← model introspection,
                         └────────┬─────────┘     config assembly
                                  │
                  ┌───────────────┼───────────────┐
                  ▼               ▼               ▼
          ┌──────────┐   ┌──────────────┐   ┌──────────────┐
          │ Segmenter│   │    Masker    │   │ProcessorOutput│
          │ (spatial)│   │ (occlusion)  │   │(preprocessing)│
          └──────────┘   └──────────────┘   └──────────────┘
                  │               │               │
                  ▼               ▼               ▼
          ┌──────────────────────────────────────────┐
          │            VisionImputer                 │
          │   coalitions → masks → forward → values  │
          └──────────────────┬───────────────────────┘
                             │
                             ▼
          ┌──────────────────────────────────────────┐
          │         VisionLanguageGame               │
          │     thin shapiq.Game adapter             │
          └──────────────────────────────────────────┘
```

**Flow**: Factory assembles Segmenter + Masker → Imputer orchestrates coalition→mask→forward
→ Game exposes `value_function` / `value_function_crossmodal` to shapiq approximators.

---

## 2. Data Protocol — `base.py`

Shared types that flow between all modules. No logic, pure data transfer.

| Type | Purpose | Key fields |
|---|---|---|
| `SpatialLayout` | Player↔pixel/token mapping | `n_players_image`, `n_players_text`, `image_size`, `patch_size`, `grid_size`, `model_type` |
| `PhysicalMask` | Concrete occlusion masks for one batch | `image_binary_mask` (N,C,H,W), `text_attention_mask` (N,L) |
| `ProcessorOutput` | Standardised HF model inputs | `pixel_values`, `input_ids`, `attention_mask`, `model_type`, `.to_dict()`, `.to(device)` |

---

## 3. Segmenters — `segmenters/`

**Contract** (`Segmenter` ABC):
- `get_layout() → SpatialLayout` — called once per image, describes spatial division
- `generate_masks(coalitions_image, coalitions_text, device) → PhysicalMask` — called per batch

**Rule**: `generate_masks()` MUST NOT access GPU ("CPU Planning, GPU Execution").
It may *receive* a `device` parameter to place output tensors, but the mask-building
logic itself runs on CPU.

**Per-strategy params** defined in `SegmenterConfig`:
- `patch` → `PatchParams` (no knobs)
- `slic` → `SlicParams(n_segments=49, compactness=10.0, sigma=0.0)`
- `gradient_guided` → `GradientGuidedParams(n_segments=None)`
- `custom_segmenter` → `CustomSegmenterParams` (no knobs)

| Segmenter | Strategy key | Description | Best for |
|---|---|---|---|
| `PatchSegmenter` | `"patch"` | Rigid ViT-aligned grid | ViT CLIP / SigLIP (default) |
| `SLICSegmenter` | `"slic"` | skimage SLIC superpixels | CNN-backbone CLIP (RN50, RN101) |
| `GradientGuidedSegmenter` | `"gradient_guided"` | Saliency-guided non-uniform layout | Research / high-saliency regions |
| `CustomSegmenter` | `"custom_segmenter"` | User-provided binary masks | Arbitrary layouts |

---

## 4. Maskers — `maskers/`

**Contract** (`Masker` ABC):
- `apply(processor_output, physical_mask) → ProcessorOutput` — applies occlusion to model inputs

**Rule**: MUST clone inputs before mutation. Never modify `processor_output` in-place.

**Per-strategy params** defined in `MaskerConfig`:
- `crossmodal_mean` → `CrossModalMeanParams` (no knobs)
- `crossmodal_blur` → `CrossModalBlurParams` (no knobs)
- `vision_mean` → `VisionMeanParams` (no knobs)
- `vision_blur` → `VisionBlurParams(sigma=3.0)`
- `text_attn` → `TextAttentionParams` (no knobs)

| Masker | Strategy key | Modifies | Mechanism |
|---|---|---|---|
| `VisionMeanMasker` | `"vision_mean"` | `pixel_values` | Multiplicative zero-out: `pixels × mask` |
| `VisionBlurMasker` | `"vision_blur"` | `pixel_values` | skimage Gaussian blur + blend (CPU) |
| `TextAttentionMasker` | `"text_attn"` | `attention_mask` | Swap attention mask for token coalitions |
| `CrossModalMeanMasker` | `"crossmodal_mean"` | both | Composite: `VisionMean` + `TextAttention` |
| `CrossModalBlurMasker` | `"crossmodal_blur"` | both | Composite: `VisionBlur` + `TextAttention` |
| `AttentionMasker` | `"attention"` | self-attention | Hooks vision encoder, injects -inf bias |

---

## 5. Cross-product — Segmenter × Masker

|  | VisionMean | VisionBlur | TextAttn | CrossModalMean | CrossModalBlur | Attention |
|---|---|---|---|---|---|---|
| **PatchSegmenter** | ✅ Image | ✅ Image | ✅ Text | ✅ VLM default | ✅ VLM | ✅ |
| **SLICSegmenter** | ✅ Image | ✅ Image | ✅ Text | ✅ VLM | ✅ VLM | ✅ |
| **GradientGuided** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Custom** | ✅ | ✅ | ✅ | ✅ | ✅ | ✅ |

---

## 6. VisionImputer — `imputer.py`

Core orchestration engine. Owns model, processor, inputs.

| Method | Input | Output | Description |
|---|---|---|---|
| `forward_1d(coalitions, batch_size)` | `(N, n_players)` bool | `(N,)` float64 | Diagonal similarity scores |
| `forward_crossmodal(img_coal, txt_coal, batch_size)` | `(N_img, K_img)`, `(N_txt, K_txt)` | `(N_img, N_txt)` | Full interaction matrix |

Internally: repeats inputs → Segmenter generates masks → Masker applies → model forward → extract outputs.
Handles batching, device placement, AMP, and crossmodal `img_bs ≠ txt_bs` edge cases.

---

## 7. VisionImputerFactory — `factory.py`

Auto-detects model type (`clip` / `siglip` / `siglip2`), extracts vision dimensions
(`image_size`, `patch_size`, `n_channels`), preprocesses text to count players,
enriches `SegmenterConfig` with model metadata, creates Segmenter + Masker, and
returns a wired `VisionImputer`.

```python
factory = VisionImputerFactory()
imputer = factory.build(model, processor, image, text)  # defaults: patch + crossmodal_mean
imputer = factory.build(model, processor, image, text,
                        segmenter_config=SegmenterConfig(strategy="slic"),
                        masker_config=MaskerConfig(strategy="vision_blur"))
```

---

## 8. VisionLanguageGame — `game.py`

Thin adapter (~75 lines) implementing `shapiq.Game`. Computes normalisation values
(empty / full coalition) and delegates `value_function` / `value_function_crossmodal`
to `VisionImputer`. Game never imports `torch`.

---

## 9. Public API

```python
# Core
from shapiq.imputer.vision import (
    VisionImputer, VisionImputerFactory, VisionLanguageGame,
)

# Segmenters
from shapiq.imputer.vision.segmenters import (
    PatchSegmenter, SLICSegmenter, GradientGuidedSegmenter, CustomSegmenter,
)

# Maskers
from shapiq.imputer.vision.maskers import (
    VisionMeanMasker, VisionBlurMasker, TextAttentionMasker,
    CrossModalMeanMasker, CrossModalBlurMasker, AttentionMasker,
)

# Abstract contracts (for custom implementations)
from shapiq.imputer.vision import Segmenter, Masker

# Config & data types
from shapiq.imputer.vision import (
    SegmenterConfig, MaskerConfig,
    PatchParams, SlicParams, GradientGuidedParams, CustomSegmenterParams,
    CrossModalMeanParams, CrossModalBlurParams,
    VisionMeanParams, VisionBlurParams, TextAttentionParams,
    SpatialLayout, PhysicalMask, ProcessorOutput,
)
```

---

## 10. Dependencies

| Dependency | Usage |
|---|---|
| `torch` | Tensor ops, model forward |
| `transformers` | HF model loading & preprocessing |
| `scikit-image` | SLIC superpixels, Gaussian blur |
| `PIL` (Pillow) | Image loading & resize |

---

## Implementation Checklist

- [x] `PatchSegmenter` + `VisionMeanMasker` on CLIP ViT-B/32
- [x] `SLICSegmenter` + `VisionBlurMasker` on CLIP-RN50
- [x] `VisionLanguageGame` `value_function()` correct shapes
- [x] `VisionImputerFactory.build()` model-type detection
- [x] All segmenter/masker abstract contract tests pass
- [x] Example notebook executes end-to-end
- [x] `GradientGuidedSegmenter` migrated and registered
- [x] `AttentionMasker` migrated and registered
- [x] Numerical equivalence with archived `ImputerFactory` verified
- [ ] Experiments migrated to `shapiq.imputer.vision` imports
