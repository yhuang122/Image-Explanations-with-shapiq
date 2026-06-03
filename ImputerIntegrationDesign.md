# shapiq Imputer Integration — Development Requirements

> **Status**: Finalised, entering development  
> **Chosen approach**: Option A (`shapiq/imputer/vision/` sub-package)  
> **Date**: 2026-06-03  
> **Lead**: PM (migration + integration), Team B (VisionBlurMasker implementation)

---

## 1. Requirements Overview

Integrate our modular Segmenter/Masker/Imputer architecture from `ImputerFactory/` + `Game/` into upstream `shapiq/imputer/` as a PR.

**Original requirement**:
> "pluggable players (patches / SLIC / custom masks) × masking strategies (mean / blur / attention / beyond)"

**PR boundary**:
- Goes upstream: abstract base classes + one complete pipeline (PatchSegmenter + VisionMeanMasker + VisionBlurMasker + VisionImputer + VisionLanguageGame) + tests + example notebook
- Stays in this project: CLIP/SigLIP model detection, experiments, FIxLIP approximator, `src/`

---

## 2. Chosen Approach: Option A — `shapiq/imputer/vision/` sub-package

Add a `vision/` sub-package under `shapiq/imputer/`, independent of the existing tabular `Imputer` hierarchy.

### Directory layout

```
shapiq/imputer/vision/
├── __init__.py                  # Public API exports
├── base.py                      # Segmenter(ABC), Masker(ABC), dataclasses
├── segmenters/
│   ├── __init__.py
│   ├── patch.py                 # PatchSegmenter (ViT rigid grid)
│   └── slic.py                  # SLICSegmenter (superpixels, skimage)
├── maskers/
│   ├── __init__.py
│   ├── vision_mean.py           # VisionMeanMasker (zero-out / mean fill)
│   ├── vision_blur.py           # VisionBlurMasker (Gaussian blur, CPU skimage)
│   ├── text_attention.py        # TextAttentionMasker (attention mask swap)
│   └── crossmodal.py            # CrossModalMeanMasker + CrossModalBlurMasker
├── imputer.py                   # VisionImputer (orchestration)
├── factory.py                   # VisionImputerFactory (assembly)
└── game.py                      # VisionLanguageGame (thin Game adapter)
```

### File mapping (this project → upstream)

| `ImputerFactory/` → | `shapiq/imputer/vision/` | Notes |
|---|---|---|
| `segmenters/base.py` ⇒ `BaseSegmenter` | `base.py` ⇒ `Segmenter` (abstract) | Renamed |
| `segmenters/patch.py` ⇒ `PatchSegmenter` | `segmenters/patch.py` ⇒ `PatchSegmenter` | Direct port |
| `segmenters/slic.py` ⇒ `SLICSegmenter` | `segmenters/slic.py` ⇒ `SLICSegmenter` | Direct port |
| `maskers/base.py` ⇒ `BaseMasker` | `base.py` ⇒ `Masker` (abstract) | Renamed |
| `maskers/vision_mean.py` | `maskers/vision_mean.py` | Direct port |
| `maskers/vision_blur.py` | `maskers/vision_blur.py` | New (CPU skimage impl.) |
| `maskers/text_attention.py` | `maskers/text_attention.py` | Direct port |
| `maskers/crossmodal_mean.py` | `maskers/crossmodal.py` ⇒ `CrossModalMeanMasker` | Direct port |
| `maskers/crossmodal_blur.py` | `maskers/crossmodal.py` ⇒ `CrossModalBlurMasker` | New composite |
| `core/imputer.py` ⇒ `ImageImputer` | `imputer.py` ⇒ `VisionImputer` | Renamed |
| `factory.py` ⇒ `ImageImputerFactory` | `factory.py` ⇒ `VisionImputerFactory` | Core logic ported |
| `data.py` (dataclasses) | `base.py` (dataclasses) | Data protocol ported |
| `Game/game_huggingface.py` ⇒ `VisionLanguageGame` | `game.py` ⇒ `VisionLanguageGame` | Direct port |

### What does NOT go into the PR

| Component | Reason |
|---|---|
| `GradientGuidedSegmenter` | Not validated |
| `AdaptiveSegmenter` / `HybridSegmenter` | Not implemented / out of scope |
| `AttentionMasker` | Stub, incomplete |
| `TorchOps` / `JAXOps` | PyTorch-only for now; JAX out of scope |
| `FIxLIP` approximator | Project-specific |
| `regression.py` | FIxLIP-specific |
| CLIP/SigLIP model auto-detection | Too specific for upstream |
| `plot.py` | Visualisation utilities |
| `src/` / `experiments/` / `analysis/` | Project legacy |

### Cross-product matrix

|  | VisionMeanMasker | VisionBlurMasker | TextAttentionMasker | CrossModalMeanMasker | CrossModalBlurMasker |
|---|---|---|---|---|---|
| **PatchSegmenter** | ✅ Image | ✅ Image | ✅ Text | ✅ VLM default | ✅ VLM |
| **SLICSegmenter** | ✅ Image | ✅ Image | ✅ Text | ✅ VLM | ✅ VLM |
| **Custom segmenter** | ✅ | ✅ | ✅ | ✅ | ✅ |

### User import paths

```python
# Core
from shapiq.imputer.vision import VisionImputer, VisionImputerFactory, VisionLanguageGame

# Segmenters
from shapiq.imputer.vision.segmenters import PatchSegmenter, SLICSegmenter

# Maskers
from shapiq.imputer.vision.maskers import (
    VisionMeanMasker, VisionBlurMasker, TextAttentionMasker,
    CrossModalMeanMasker, CrossModalBlurMasker,
)

# Abstract contracts (for custom implementations)
from shapiq.imputer.vision import Segmenter, Masker

# Data types
from shapiq.imputer.vision import (
    SegmenterConfig, MaskerConfig,
    SpatialLayout, PhysicalMask, ProcessorOutput,
)
```

---

## 3. Abstract Contracts

### Segmenter

```python
class Segmenter(ABC):
    @abstractmethod
    def get_layout(self) -> SpatialLayout: ...
    @abstractmethod
    def generate_masks(self, coalitions_image: np.ndarray,
                       coalitions_text: np.ndarray) -> PhysicalMask: ...
```

**Rule**: `generate_masks()` MUST NOT access the GPU ("CPU Planning, GPU Execution").

### Masker

```python
class Masker(ABC):
    @abstractmethod
    def apply(self, processor_output: ProcessorOutput,
              physical_mask: PhysicalMask) -> ProcessorOutput: ...
```

**Rule**: MUST clone inputs before mutation. Never modify in-place.

### Data types

| Type | Description | Fields |
|---|---|---|
| `SpatialLayout` | Immutable spatial metadata | `n_players_image`, `n_players_text`, `grid_size`, `image_size`, `patch_size` |
| `PhysicalMask` | Concrete occlusion masks | `image_binary_mask: Tensor (N,C,H,W)`, `text_attention_mask: Tensor (M,L)` |
| `ProcessorOutput` | Standardised model inputs | `pixel_values`, `input_ids`, `attention_mask` + `.to_dict()` |
| `SegmenterConfig` | Segmenter configuration | `strategy`, per-strategy params, model metadata |
| `MaskerConfig` | Masker configuration | `strategy`, per-strategy params |

---

## 4. VisionBlurMasker Design

CPU-only (Phase 1). GPU conv2d optimisation is a future concern.

```python
# Core logic: skimage.filters.gaussian on CPU numpy
for batch_idx in range(N):
    for c in range(C):
        blurred[batch_idx, c] = gaussian_blur(im_np[batch_idx, c], sigma=sigma)
# Blend
blended = im_np * mask_np + blurred * (1.0 - mask_np)
```

- `sigma` constructor parameter, default 3.0
- `CrossModalBlurMasker` composes `VisionBlurMasker` + `TextAttentionMasker`

---

## 5. Upstream Dependencies

| Dependency | Status | Notes |
|---|---|---|
| `torch` | Already in shapiq | Used by `shapiq.explainer.nn` |
| `transformers` | **New** | HuggingFace model loading |
| `scikit-image` | **New** | SLIC superpixels, Gaussian blur |
| `PIL` (Pillow) | Already in shapiq | Image loading |

**No changes to existing shapiq code required.** The vision sub-package is purely additive.

---

## 6. Acceptance Criteria

| # | Criterion | Method |
|---|---|---|
| AC1 | `PatchSegmenter` + `VisionMeanMasker` pipeline runs on CLIP ViT-B/32 | Equivalent to `example.ipynb` |
| AC2 | `SLICSegmenter` + `VisionBlurMasker` pipeline runs on CLIP-RN50 | Equivalent to SLIC example |
| AC3 | `VisionLanguageGame(Game)` `value_function()` returns correct shapes | Test assertions |
| AC4 | `VisionImputerFactory.build()` correctly infers model type and assembles components | Test assertions |
| AC5 | All abstract contract tests pass for every segmenter and masker | `pytest tests/imputer/vision/` |
| AC6 | Example notebook executes without errors | End-to-end run |

---

## Appendix

### A. Solution comparison

| Approach | Description | Decision |
|---|---|---|
| **Option A** | New `shapiq/imputer/vision/` sub-package, independent of tabular Imputer | ✅ **Chosen** |
| **Option B** | Refactor `Imputer` base class to accept vision parameters | ❌ High backward-compat risk |
| **Option C** | Place `VisionLanguageGame` under `shapiq/games/`, Segmenter/Masker unexposed | ❌ Does not meet modularity goal |

### B. Option B summary

Refactor `Imputer.__init__` to accept optional domain-specific parameters (`model, processor, image, text` in place of `data, x`), and add `_segment()` / `_occlude()` abstract methods to the base. Pro: unified hierarchy. Con: backward-incompatible, semantic mismatch between tabular imputation and vision masking, longer PR cycle.

### C. Option C summary

Place `VisionLanguageGame` directly under `shapiq/explainer/nn/games/` (alongside existing KNN/TNN Games), without exposing Segmenter/Masker interfaces. Minimal upstream changes but fails to achieve the stated goal of "making shapiq modular with pluggable segmenters and maskers."
