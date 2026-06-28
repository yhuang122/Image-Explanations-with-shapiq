# ImageImputer — Architecture & Design

> Last updated: 2026-06-02

## Module Responsibilities & Boundaries

> Each module has a single, well-defined responsibility.  
> Teams must not cross these boundaries without prior discussion with the PM.

### Segmenter
**Sole responsibility**: Spatial division — define which pixels/tokens belong to which player.

- **Owns**: layout metadata (`SpatialLayout`), the algorithm for converting coalitions → pixel/token masks
- **Does NOT own**: how occlusion is applied (that's the Masker), model forward pass, batching logic
- **Contract**:
  - `get_layout()` → returns `SpatialLayout` (called once)
  - `generate_masks(coalitions_image, coalitions_text)` → returns `PhysicalMask` (called per batch)
- **Must NOT**: access `model`, call `processor`, mutate tensors on GPU inside `generate_masks`
- **Allowed to vary**: grid shape, player count, player-to-pixel mapping, mask generation algorithm
- **Examples**: `PatchSegmenter` (rigid grid), `SLICSegmenter` (perceptual superpixels), `GradientGuidedSegmenter` (gradient-based, future)

### Masker
**Sole responsibility**: Feature occlusion — apply the `PhysicalMask` to `ProcessorOutput`.

- **Owns**: the occlusion strategy (multiplicative, attention injection, etc.)
- **Does NOT own**: mask generation, model forward pass, batching logic
- **Contract**:
  - `apply(processor_output, physical_mask)` → returns `ProcessorOutput` (modified)
- **Must NOT**: access `model`, call `processor`, generate masks from coalitions
- **Must**: clone inputs before mutation (never mutate the original)
- **Examples**: `VisionMeanMasker` (pixel multiply), `TextAttentionMasker` (attention swap), `CrossModalMeanMasker` (composite of the two), `AttentionMasker` (negative-infinity injection)

### ImageImputer
**Sole responsibility**: Orchestration — coordinate Segmenter → Masker → Model forward.

- **Owns**: model, processor, raw inputs (`input_image`/`input_text`), preprocessed `inputs_original`, batching loop, device management
- **Does NOT own**: segmentation algorithm, occlusion strategy
- **Contract**:
  - `forward_1d(coalitions, batch_size)` → `np.array` (diagonal logits)
  - `forward_crossmodal(coalitions_img, coalitions_txt, batch_size)` → `np.array` (full matrix)
- **Flow it orchestrates**: `Segmenter.generate_masks → Masker.apply → model(**inputs) → extract logits`
- **Must**: handle device placement, batch iteration, edge cases (txt_bs ≠ img_bs)
- **Must NOT**: implement mask generation logic directly, call `processor` gratuitously

### ImageImputerFactory
**Sole responsibility**: Assembly — inspect the model and wire up the correct components.

- **Owns**: component selection logic, model introspection, one-time preprocessing
- **Does NOT own**: segmentation, occlusion, forward pass, batching
- **Contract**:
  - `build(model, processor, input_image, input_text, segmenter_config=..., masker_config=...)` → `ImageImputer`
- **Must**: infer model type, count text players, create appropriate Segmenter+Masker
- **Must NOT**: run model forward, generate masks, apply occlusion

### VisionLanguageGame (in `Game/`)
**Sole responsibility**: shapiq adapter — provide the `value_function` interface that shapiq's approximator expects.

- **Owns**: normalization values (`empty_value`/`full_value`), batch_size, backward-compat delegation properties
- **Does NOT own**: model, processor, masking, batching
- **Contract**:
  - `value_function(coalitions)` → delegating to `imputer.forward_1d`
  - `value_function_crossmodal(img, txt)` → delegating to `imputer.forward_crossmodal`
- **Must NOT**: import `torch`, call `model` directly, generate masks, implement value_function logic

### Data Types (in `ImputerFactory/data.py`)
**Sole responsibility**: Universal data protocol — define the shapes and semantics of objects passed between modules.

- `SegmenterConfig`: typed segmenter configuration. Caller provides `strategy` + per-strategy params (`PatchParams`, `SlicParams`, `GradientGuidedParams`). Factory populates model metadata (`model_type`, `image_size`, `patch_size`, `n_channels`, `grid_size`, `n_players_image`, `n_players_text`, `text_total_length`). Consumed by Segmenters.
- `MaskerConfig`: typed masker configuration. Caller provides `strategy` + per-strategy params (`CrossModalMeanParams`, `VisionBlurParams`, `VisionMeanParams`, `TextAttentionParams`). Consumed by Maskers.
- `SpatialLayout`: immutable metadata (produced by Segmenter, consumed by Imputer)
- `PhysicalMask`: concrete tensor masks (produced by Segmenter/imputer translation, consumed by Masker)
- `ProcessorOutput`: standardized model inputs (produced by Factory, consumed by Masker and Imputer)

### Boundary Rules

| Rule | Rationale |
|---|---|
| Only the Imputer touches `model` | Prevents scattered `model(**inputs)` calls |
| Only the Imputer owns `processor` and raw `input_image`/`input_text` | Single source of truth for preprocessing |
| `SegmenterConfig` / `MaskerConfig` are read-only after Factory enriches them | Typed configs carry strategy + params + metadata; no loose dicts |
| Segmenter never accesses GPU inside `generate_masks` | "CPU Planning, GPU Execution": masks generated once, applied in bulk |
| Masker clones inputs before modifying | Prevents mutation bugs across batch iterations |
| Game never imports `torch` | Keeps the adapter pure; all tensor ops live in ImputerFactory |

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

---

## Key Design Decisions

1. **"CPU Planning, GPU Execution"**: Segmenters produce integer index maps once on CPU (via skimage). Thousands of coalition→mask translations happen purely on GPU via native tensor ops.

2. **Typed Component Configuration**: `SegmenterConfig` and `MaskerConfig` carry typed per-strategy params alongside model metadata. The Factory populates model metadata into `SegmenterConfig` during `build()`; callers only provide strategy + params. No loose dicts, full IDE autocomplete, and per-strategy params are self-documenting via `help(SlicParams)`.

3. **Imputer owns the inputs**: `ImageImputer` stores `inputs_original` (ProcessorOutput), `inputs_raw` (HF dict for `.tokens()`), and `input_image`/`input_text` (for crossmodal edge cases where batch sizes diverge).

4. **Game is a thin shell**: `VisionLanguageGame` delegates all masking/batching/model-forward to the Imputer. It only handles shapiq scheduling (normalization values, player counts).

---

## Current Implementation Details

### `ImputerFactory/data.py`
Six dataclasses serve as the universal data protocol:
- **`SegmenterConfig`**: Typed segmenter configuration. Carries `strategy` ("patch", "slic", "gradient_guided") + per-strategy params (`PatchParams`, `SlicParams`, `GradientGuidedParams`) + Factory-populated model metadata. Defaults to "patch".
- **`MaskerConfig`**: Typed masker configuration. Carries `strategy` ("crossmodal_mean", "crossmodal_blur", "vision_mean", "vision_blur", "text_attn") + per-strategy params. Defaults to "crossmodal_mean".
- **`SpatialLayout`**: Immutable metadata describing the spatial division. Produced once by Segmenter, consumed by Imputer.
- **`PhysicalMask`**: Concrete tensor masks. `image_binary_mask` (N, C, H, W) float + `text_attention_mask` (N, L) int.
- **`ProcessorOutput`**: Wraps `pixel_values`, `input_ids`, `attention_mask` with a `to_dict()` for model forwarding.

### `ImputerFactory/segmenters/patch.py` — PatchSegmenter
- Constructor: `PatchSegmenter(config: SegmenterConfig)` — receives typed config with model metadata
- All model metadata (image_size, patch_size, etc.) read from `config`, no direct model access
- Pre-computes `SpatialLayout` at init (is_stateful=False)
- `generate_masks()` converts coalition arrays → `PhysicalMask`
- Image: expand patch-level booleans → `patch_size×patch_size` blocks → (N, C, H, W)
- Text: handles CLIP (BOS/EOS wrapping) vs SigLIP (right-padding) mask formats

### `ImputerFactory/segmenters/slic.py` — SLICSegmenter
- **Scope**: CNN models — uses perceptual superpixel segmentation to preserve natural object boundaries
- Constructor: `SLICSegmenter(config: SegmenterConfig, image_array)` — `image_array` passed by Factory
- **Algorithm**: `skimage.segmentation.slic` on CPU during `get_layout()` (one-time "CPU Planning")
  - `n_segments`, `compactness`, `sigma` read from `config.slic` (typed `SlicParams`)
  - Produces a 2D integer index map: each pixel → player ID
- **GPU path**: The index map is uploaded to GPU once; `generate_masks()` uses `torch.where` / scatter to expand coalition bits into pixel-level binary masks
- **Why it matters for CNNs**: Rigid-grid patches (PatchSegmenter) create Out-of-Distribution artifacts for CNN-based models because they introduce sharp artificial edges. SLIC superpixels follow natural image contours, preserving the input distribution.

### `ImputerFactory/maskers/vision_mean.py` — VisionMeanMasker
- **Sole responsibility**: Pure image occlusion via multiplicative binary mask
- Registered as `"vision_mean"` in the masker registry
- `apply()`: `pixel_values *= image_binary_mask` → returns `ProcessorOutput` with only `pixel_values` modified
- Never touches `input_ids` or `attention_mask` — safe to compose
- Serves PatchSegmenter (ViT), SLICSegmenter (CNN), or any future image-only segmenter

### `ImputerFactory/maskers/text_attention.py` — TextAttentionMasker
- **Sole responsibility**: Pure text occlusion via `attention_mask` replacement
- Registered as `"text_attn"` in the masker registry
- `apply()`: replaces `attention_mask` with coalition-derived mask → returns `ProcessorOutput` with only `attention_mask` modified
- Never touches `pixel_values` — safe to compose
- Serves text-only models or cross-modal pipelines

### `ImputerFactory/maskers/crossmodal_composite.py` — CrossModalMeanMasker
- **Sole responsibility**: Composite Pattern — orchestrates VisionMeanMasker + TextAttentionMasker
- Registered as `"crossmodal_mean"`; default masker for VLMs (CLIP/SigLIP)
- Internally instantiates both atomic maskers; `apply()` delegates image → Vision, then text → Text
- Owns no low-level tensor math — purely an orchestration layer

### `ImputerFactory/maskers/crossmodal_blur.py` — CrossModalBlurMasker
- **Sole responsibility**: Composite Pattern with Gaussian-blur image occlusion
- Registered as `"crossmodal_blur"`
- Text side: delegates to TextAttentionMasker (`"text_attn"`)
- Image side: delegates to VisionBlurMasker (`"vision_blur"`) — CPU Gaussian blur + blend

### `ImputerFactory/core/imputer.py` — ImageImputer
- Constructor: receives `use_amp` (bool) directly; all spatial metadata derived from `segmenter.get_layout()` (`SpatialLayout`)
- **`forward_1d(coalitions, batch_size)`**: Splits coalitions → generates masks → batches → masks → model → extracts diagonal
- **`forward_crossmodal(coalitions_img, coalitions_txt, batch_size)`**: Double loop (image outer, text inner). Edge case: when txt_bs ≠ img_bs, re-processes via `_preprocess_batch()` using stored `input_image`/`input_text`
- **`_model_forward()`**: Auto-detects model device, moves inputs before forward; supports `use_amp` for fp16 autocast on CUDA
- Stores: `model`, `processor`, `segmenter`, `masker`, `layout`, `inputs_original`, `inputs_raw`, `input_image`, `input_text`, `use_amp`

### `ImputerFactory/factory.py` — ImageImputerFactory
- `build(model, processor, input_image, input_text, segmenter_config=None, masker_config=None, use_amp=False)`:
  1. Infers model type (clip/siglip/siglip2)
  2. Extracts model dimensions and computes derived values (grid_size, n_players_image)
  3. Preprocesses once to determine `n_players_text` + `text_total_length`
  4. **Enriches `SegmenterConfig`** with model metadata (caller only supplies strategy + params)
  5. Creates segmenter via per-strategy dispatch:
     - `"patch"`: `cls(config=segmenter_config)`
     - `"slic"`: `cls(config=segmenter_config, image_array=input_image)`
     - `"gradient_guided"`: `cls(config=segmenter_config, model=model, processor=processor, image=input_image, text=input_text)`
  6. Creates masker via `_create_masker(masker_config)`
  7. Wires `ProcessorOutput` + raw dict + segmenter + masker into `ImageImputer`

### `Game/game_huggingface.py` — VisionLanguageGame
- Constructor: `VisionLanguageGame(imputer, batch_size=64, verbose=False)`
- `n_players_image` / `n_players_text` from imputer layout
- `inputs` / `processor` properties delegate to imputer (backward compat)
- `value_function()` → `imputer.forward_1d()`
- `value_function_crossmodal()` → `imputer.forward_crossmodal()`

---

---

## Evaluation Experiments

> **Ownership**: Team B — Feature Development. A group provides validation via A2 comparison harness.

### Insertion / Deletion Curve (`experiments/insertion_deletion.py`)

**Purpose**: Evaluate the faithfulness of attribution methods by measuring how the model's similarity score changes as features are progressively removed (deletion) or added (insertion), ordered by their attributed importance.

**How it works**:
1. Load pre-computed `InteractionValues` from `explain_mscoco.py` / `explain_mscoco_siglip.py`
2. Sort players by attribution value (high → low for MIF, low → high for LIF)
3. Generate a sequence of coalitions that remove one more player at each step
4. Run `game.value_function()` on all coalitions, producing a similarity curve
5. Normalize the curve to [0, 1] and compute AID (Area between Insertion and Deletion curves)

**Curve layout**:
- **Y-axis (left)**: Prediction change — normalized model similarity score
- **X-axis (bottom)**: Fraction of input kept — k / (n_img + n_txt)
- **Deletion curve**: starts at full input, each step removes the next player → score drops
- **Insertion curve**: starts at empty input, each step adds the next player → score rises
- **AID (Area between curves)**: Δ between insertion and deletion curves; higher = more faithful attribution

**Single-sample test result** (dog_and_hydrant, CLIP ViT-B/32):

| Order | AID | Interpretation |
|---|---|---|
| 1 (1st-order only) | -0.22 | Poor: LIF drops faster than MIF — attributions misordered |
| 2 (with 2nd-order interactions) | +1.13 | Good: MIF drops monotonically, LIF stays high |

**Migration plan** → ImputerFactory:
- Replace `src.game_huggingface.VisionLanguageGame(model, processor, ...)` with `Game.game_huggingface.VisionLanguageGame(imputer, ...)`
- `game.value_function()` is unchanged in API — remains the same contract
- In crossmodal mode with cliques (n_players > 100), ensure `imputer.forward_1d()` handles the coalition batches correctly
- Validation: compare AID values before/after migration (±1e-4 tolerance)

---

## Known Issues

- **Crossmodal edge-case processor calls**: When `budget_image % batch_size ≠ 0` or `budget_text % batch_size ≠ 0`, the last image and/or text batch have incomplete sizes (e.g., `img_bs=15, txt_bs=51` for `batch_size=64, budget_image=4559, budget_text=115`). The 2 (img batches) × 2 (text batches) = 4 combinations yield 3 cases where `img_bs ≠ txt_bs`. In those cases `_preprocess_batch()` must re-invoke the HF processor to create inputs with matching batch dimensions. The original `src` code has the same behavior (it calls `processor_function` directly in the equivalent branches), so this is not a regression — it is inherent to the double-loop crossmodal design. Total extra calls per `forward_crossmodal`: at most 3 (~2 ms each, negligible).

---

## API: Typed Component Configuration

The `build()` API uses typed `SegmenterConfig` and `MaskerConfig` dataclasses instead of string-based `segmenter`/`segmenter_kwargs`/`masker` parameters. `ImputerConfig` has been removed — model metadata flows through `SegmenterConfig` (populated by Factory) and spatial metadata through `SpatialLayout`.

### Usage

```python
# Defaults (backward-compatible)
imputer = factory.build(model, processor, img, txt)

# Custom segmenter
from ImputerFactory import SegmenterConfig, SlicParams
seg_cfg = SegmenterConfig(strategy="slic", slic=SlicParams(n_segments=60))
imputer = factory.build(model, processor, img, txt, segmenter_config=seg_cfg)

# Custom masker
from ImputerFactory import MaskerConfig
msk_cfg = MaskerConfig(strategy="crossmodal_blur")
imputer = factory.build(model, processor, img, txt, masker_config=msk_cfg)
```

### Dataclass summary

| Class | Fields |
|---|---|
| `SegmenterConfig` | `strategy` ("patch"/"slic"/"gradient_guided"), `patch`, `slic`, `gradient_guided` + Factory-populated model metadata |
| `MaskerConfig` | `strategy` ("crossmodal_mean"/"crossmodal_blur"/"vision_mean"/"vision_blur"/"text_attn"), `crossmodal_mean`, `vision_blur`, `vision_mean`, `text_attn` |
| `PatchParams` | (empty) |
| `SlicParams` | `n_segments` (49), `compactness` (10.0), `sigma` (0.0) |
| `GradientGuidedParams` | `n_segments` (None → auto) |
| `CrossModalMeanParams` | (empty) |
| `VisionBlurParams` | `sigma` (3.0) |
| `VisionMeanParams` | (empty) |
| `TextAttentionParams` | (empty) |

### Factory flow

```
build(model, processor, image, text, segmenter_config=..., masker_config=..., use_amp=...)
  │
  ├─ 1. Infer model type (clip/siglip/siglip2)
  ├─ 2. Extract model dimensions (image_size, patch_size, n_channels)
  ├─ 3. Preprocess once → n_players_text, text_total_length
  ├─ 4. Enrich segmenter_config with model metadata
  ├─ 5. Create segmenter via per-strategy dispatch
  ├─ 6. Sync segmenter_config.n_players_image from layout
  ├─ 7. Create masker: cls(config=masker_config)
  └─ 8. Assemble ImageImputer (reads spatial metadata from layout, use_amp directly)
```

### Masker registry keys

| Key | Class |
|---|---|
| `"vision_mean"` | VisionMeanMasker |
| `"text_attn"` | TextAttentionMasker |
| `"crossmodal_mean"` | CrossModalMeanMasker (default) |
| `"crossmodal_blur"` | CrossModalBlurMasker |
| `"vision_blur"` | VisionBlurMasker |
| `"attention"` | AttentionMasker (stub) |
