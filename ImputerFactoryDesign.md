# ImageImputer — Architecture & Design

> Last updated: 2026-05-13

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
- **Examples**: `PatchSegmenter` (rigid grid), `SLICSegmenter` (perceptual superpixels), `GradientGuidedSegmenter` (gradient-based, future), `AdaptiveSegmenter` (score-driven, future)

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
  - `build(model, processor, input_image, input_text, segmenter=..., masker=...)` → `ImageImputer`
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

- `ImputerConfig`: shared read-only configuration (produced by Factory, consumed by all modules). Contains model metadata (`model_type`, `image_size`, `patch_size`, `n_channels`, `n_players_image`, `n_players_text`, `grid_size`, `text_total_length`, `use_amp`), component selection (`segmenter`, `masker`), and `segmenter_kwargs` for variable block sizing. (Planned: component selection and params will move to `SegmenterConfig` / `MaskerConfig` — see "Future API Evolution".)
- `SpatialLayout`: immutable metadata (produced by Segmenter, consumed by Imputer)
- `PhysicalMask`: concrete tensor masks (produced by Segmenter/imputer translation, consumed by Masker)
- `ProcessorOutput`: standardized model inputs (produced by Factory, consumed by Masker and Imputer)

### Boundary Rules

| Rule | Rationale |
|---|---|
| Only the Imputer touches `model` | Prevents scattered `model(**inputs)` calls |
| Only the Imputer owns `processor` and raw `input_image`/`input_text` | Single source of truth for preprocessing |
| `ImputerConfig` is read-only after Factory creates it | Ensures all components share a consistent view of model metadata |
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

2. **ImputerConfig as shared truth**: `ImputerConfig` is created once by the Factory and shared read-only across Segmenter, Masker, and Imputer. This eliminates scattered model introspection. `segmenter_kwargs` carries variable block-size parameters from Factory → Segmenter. (Planned: typed `SegmenterConfig` / `MaskerConfig` will replace this — see "Future API Evolution".)

3. **Imputer owns the inputs**: `ImageImputer` stores `inputs_original` (ProcessorOutput), `inputs_raw` (HF dict for `.tokens()`), and `input_image`/`input_text` (for crossmodal edge cases where batch sizes diverge).

4. **Game is a thin shell**: `VisionLanguageGame` delegates all masking/batching/model-forward to the Imputer. It only handles shapiq scheduling (normalization values, player counts).

---

## Current Implementation Details

### `ImputerFactory/data.py`
Four dataclasses serve as the universal data protocol:
- **`ImputerConfig`**: Read-only configuration produced by Factory. Contains `model_type`, `image_size`, `patch_size`, `n_channels`, `n_players_image`, `n_players_text`, `grid_size`, `text_total_length`, `segmenter`, `masker`, and `segmenter_kwargs` (forwarded to Segmenter for variable block sizing). Shared across all components.
- **`SpatialLayout`**: Immutable metadata describing the spatial division. Produced once by Segmenter, consumed by Imputer.
- **`PhysicalMask`**: Concrete tensor masks. `image_binary_mask` (N, C, H, W) float + `text_attention_mask` (N, L) int.
- **`ProcessorOutput`**: Wraps `pixel_values`, `input_ids`, `attention_mask` with a `to_dict()` for model forwarding.

### `ImputerFactory/segmenters/patch.py` — PatchSegmenter
- Constructor: `PatchSegmenter(config: ImputerConfig)` — receives shared config
- All model metadata (image_size, patch_size, etc.) read from `config`, no direct model access
- Pre-computes `SpatialLayout` at init (is_stateful=False)
- `generate_masks()` converts coalition arrays → `PhysicalMask`
- Image: expand patch-level booleans → `patch_size×patch_size` blocks → (N, C, H, W)
- Text: handles CLIP (BOS/EOS wrapping) vs SigLIP (right-padding) mask formats

### `ImputerFactory/segmenters/slic.py` — SLICSegmenter
- **Scope**: CNN models — uses perceptual superpixel segmentation to preserve natural object boundaries
- Constructor: `SLICSegmenter(config: ImputerConfig)`
- **Algorithm**: `skimage.segmentation.slic` on CPU during `get_layout()` (one-time "CPU Planning")
  - `n_segments` controlled via `config.segmenter_kwargs`
  - Produces a 2D integer index map: each pixel → player ID
- **GPU path**: The index map is uploaded to GPU once; `generate_masks()` uses `torch.where` / scatter to expand coalition bits into pixel-level binary masks
- **Why it matters for CNNs**: Rigid-grid patches (PatchSegmenter) create Out-of-Distribution artifacts for CNN-based models because they introduce sharp artificial edges. SLIC superpixels follow natural image contours, preserving the input distribution.

### `ImputerFactory/maskers/vision_mean.py` — VisionMeanMasker
- **Sole responsibility**: Pure image occlusion via multiplicative binary mask
- `apply()`: `pixel_values *= image_binary_mask` → returns `ProcessorOutput` with only `pixel_values` modified
- Never touches `input_ids` or `attention_mask` — safe to compose
- Serves PatchSegmenter (ViT), SLICSegmenter (CNN), or any future image-only segmenter

### `ImputerFactory/maskers/text_attention.py` — TextAttentionMasker
- **Sole responsibility**: Pure text occlusion via `attention_mask` replacement
- `apply()`: replaces `attention_mask` with coalition-derived mask → returns `ProcessorOutput` with only `attention_mask` modified
- Never touches `pixel_values` — safe to compose
- Serves text-only models or cross-modal pipelines

### `ImputerFactory/maskers/crossmodal_composite.py` — CrossModalMeanMasker
- **Sole responsibility**: Composite Pattern — orchestrates VisionMeanMasker + TextAttentionMasker
- Internally instantiates both atomic maskers; `apply()` delegates image → Vision, then text → Text
- Owns no low-level tensor math — purely an orchestration layer
- The default masker for VLMs (CLIP/SigLIP)

### `ImputerFactory/core/imputer.py` — ImageImputer
- Constructor: receives `config: ImputerConfig` — model metadata derived from config, not from `model` directly
- **`forward_1d(coalitions, batch_size)`**: Splits coalitions → generates masks → batches → masks → model → extracts diagonal
- **`forward_crossmodal(coalitions_img, coalitions_txt, batch_size)`**: Double loop (image outer, text inner). Edge case: when txt_bs ≠ img_bs, re-processes via `_preprocess_batch()` using stored `input_image`/`input_text`
- **`_model_forward()`**: Auto-detects model device, moves inputs before forward
- Stores: `config`, `inputs_original`, `inputs_raw`, `input_image`, `input_text`, `model`, `processor`, `segmenter`, `masker`, `layout`

### `ImputerFactory/factory.py` — ImageImputerFactory
- `build(model, processor, input_image, input_text, segmenter=None, masker=None)`:
  1. Infers model type (clip/siglip/siglip2)
  2. Extracts model dimensions and computes derived values (grid_size, n_players_image)
  3. Preprocesses once to determine `n_players_text` + `text_total_length`
  4. **Builds `ImputerConfig`** — single source of truth for all metadata, segmenter/masker selection, and `segmenter_kwargs`
  5. Creates segmenter via `_create_segmenter(config)` — config flows into Segmenter
  6. Creates masker via `_create_masker(config)` — config flows into Masker
  7. Wires `ProcessorOutput` + raw dict + config + raw image/text into `ImageImputer`

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
## Future API Evolution: Typed Component Configuration

> Planned for Phase 2. Not yet implemented.
> Target: Replace string-based `segmenter`/`segmenter_kwargs`/`masker` parameters
> with typed `SegmenterConfig` and `MaskerConfig` dataclasses.

### Motivation

The current `build()` API uses three loosely-typed parameters to describe component selection:

```python
def build(self, model, processor, input_image, input_text,
          segmenter: Optional[str] = None,         # "patch" / "slic" / "gradient_guided"
          segmenter_kwargs: Optional[dict] = None,  # weak dict: keys discovered at runtime
          masker: Optional[str] = None,             # "crossmodal_mean" / "vision" / "text"
          ...)
```

Three problems this creates:

1. **`segmenter_kwargs` is a type hole.** IDE autocomplete cannot suggest keys, mypy cannot catch typos (`n_segments` vs `n_segements`), and per-strategy parameters mix in one flat dict (`image_array` for SLIC coexists with `model` for gradient-guided). The only way to discover available keys is to read the segmenter constructor source code.

2. **Component parameters mixed with model metadata in `ImputerConfig`.** Today `ImputerConfig` carries two categories of data with different origins: model metadata derived from introspection (`image_size`, `patch_size`, `model_type`, etc.) **and** user-provided component selection (`segmenter`, `masker`) plus a dict of strategy-specific params (`segmenter_kwargs`). These are conceptually separate: the Factory produces model metadata, the caller provides component choices. Mixing them in one dataclass makes it harder to reason about what the Factory owns vs what the caller controls. (Note: this is a design smell, not a boundary violation — the Factory still constructs `ImputerConfig` internally in both the current and proposed design, so the "ImputerConfig is produced by Factory" rule is satisfied either way.)

3. **No migration path for new parameters.** Every new segmenter adds more undocumented keys to the same `dict`. Callers must grep the segmenter constructor to learn what keys are accepted.

### Proposed solution: `SegmenterConfig` + `MaskerConfig`

Add new configuration dataclasses to `ImputerFactory/data.py`:
- `SegmenterConfig`: carries strategy, typed per-strategy params, AND model metadata (populated by the Factory). The segmenter constructor receives only this object.
- `MaskerConfig`: carries strategy + typed per-strategy params (currently empty, future-proofed for `AttentionMasker`).
- `ImputerConfig` is **removed** — it becomes redundant when model metadata flows through `SegmenterConfig` and spatial metadata through `SpatialLayout`.

`build()` accepts `segmenter_config` and `masker_config` instead of `segmenter`/`segmenter_kwargs`/`masker` strings.

### Planned dataclass design

```python
# ── Segmenter parameter types (one per strategy) ─────────────────────

@dataclass
class PatchParams:
    """Rigid-grid patch segmenter. No configurable parameters."""
    pass


@dataclass
class SlicParams:
    """SLIC superpixel segmentation parameters."""
    n_segments: int = 49
    compactness: float = 10.0
    sigma: float = 0.0


@dataclass
class GradientGuidedParams:
    """Gradient-guided saliency segmentation."""
    n_segments: Optional[int] = None  # None → derive from grid_size


@dataclass
class SegmenterConfig:
    """
    Complete configuration for a Segmenter. Accepts only this one object.

    Fields are in two categories:
      - User-provided: strategy + per-strategy params (patch/slic/gradient_guided).
      - Factory-populated: model metadata (image_size, patch_size, etc.).
        The user does not need to supply these — the Factory fills them
        during build() based on model introspection.
    """
    # ── User-provided ─────────────────────────────────────────────
    strategy: Literal["auto", "patch", "slic", "gradient_guided"] = "auto"
    patch: PatchParams = field(default_factory=PatchParams)
    slic: SlicParams = field(default_factory=SlicParams)
    gradient_guided: GradientGuidedParams = field(default_factory=GradientGuidedParams)

    # ── Factory-populated (model metadata) ────────────────────────
    model_type: str = ""
    image_size: int = 0
    patch_size: int = 0
    n_channels: int = 3
    grid_size: int = 0
    n_players_image: int = 0
    n_players_text: int = 0
    text_total_length: int = 0

    @property
    def active_params(self):
        """Return the params dataclass for the active strategy."""
        return getattr(self, self.strategy, None)


# ── Masker parameter types (one per strategy) ─────────────────────────

@dataclass
class CrossModalMeanParams:
    """Cross-modal occlusion (composite: vision-mean + text-attention). No configurable parameters."""
    pass


@dataclass
class VisionMeanParams:
    """Pure image occlusion via multiplicative binary mask. No configurable parameters."""
    pass


@dataclass
class TextAttentionParams:
    """Pure text occlusion via attention_mask replacement. No configurable parameters."""
    pass


@dataclass
class MaskerConfig:
    """
    Complete configuration for a Masker. Accepts only this one object.

    Fields in two categories:
      - User-provided: strategy + per-strategy params.
      - Factory may enrich with model metadata if needed (currently unused).

    Future maskers (e.g. AttentionMasker) will add their own params
    dataclass here and extend `strategy` with a new literal value.
    """
    strategy: Literal["auto", "crossmodal_mean", "vision", "text"] = "auto"
    crossmodal_mean: CrossModalMeanParams = field(default_factory=CrossModalMeanParams)
    vision: VisionMeanParams = field(default_factory=VisionMeanParams)
    text: TextAttentionParams = field(default_factory=TextAttentionParams)

    @property
    def active_params(self):
        """Return the params dataclass for the active strategy."""
        return getattr(self, self.strategy, None)
```

### New `build()` signature

```python
from ImputerFactory.data import SegmenterConfig, MaskerConfig

class ImageImputerFactory:
    def build(
        self,
        model: Any,
        processor: Any,
        input_image: Any,
        input_text: str,
        segmenter_config: Optional[SegmenterConfig] = None,  # ← replaces segmenter + segmenter_kwargs
        masker_config: Optional[MaskerConfig] = None,         # ← replaces masker
        use_amp: bool = False,
    ) -> ImageImputer:
```

The old `segmenter`, `segmenter_kwargs`, and `masker` parameters are removed entirely. `None` means "auto-select" (preserving current default behavior).

**Before / After comparison:**

| Before | After |
|---|---|
| `build(model, proc, img, txt)` | `build(model, proc, img, txt)` — unchanged |
| `build(..., segmenter="slic")` | `build(..., segmenter_config=SegmenterConfig(strategy="slic"))` |
| `build(..., segmenter="slic", segmenter_kwargs={"n_segments": 60})` | `build(..., segmenter_config=SegmenterConfig(strategy="slic", slic=SlicParams(n_segments=60)))` |
| `build(..., segmenter="gradient_guided")` | `build(..., segmenter_config=SegmenterConfig(strategy="gradient_guided"))` |
| `build(..., masker="vision")` | `build(..., masker_config=MaskerConfig(strategy="vision"))` |
| (future) `AttentionMasker(temperature=0.5)` | `build(..., masker_config=MaskerConfig(strategy="attention", attention=AttentionParams(temperature=0.5)))` |

### On the necessity of `ImputerConfig`

| Consumer | What it reads from `ImputerConfig` |
|---|---|
| Segmenter | `image_size`, `patch_size`, `n_channels`, `model_type`, `n_players_text`, `text_total_length`, `grid_size`, `n_players_image` + `segmenter_kwargs` |
| Masker | nothing (currently parameterless) |
| ImageImputer | `image_size`, `patch_size`, `n_channels`, `grid_size`, `model_type`, `use_amp` |

After the refactoring:
- Segmenter receives a **complete** `SegmenterConfig` that carries model metadata + strategy + typed params.
- Masker receives `MaskerConfig` (currently just strategy; extensible for future params).
- ImageImputer can read all spatial metadata from `SpatialLayout` (already produced by `segmenter.get_layout()`), which contains `image_size`, `patch_size`, `n_channels`, `grid_size`, `model_type`, `n_players_image`, `n_players_text`, `text_total_length`. The only remaining field is `use_amp`, which can be stored directly on ImageImputer as a simple bool.

At that point every consumer gets what it needs without `ImputerConfig`. The class becomes a middleman: the Factory produces metadata, writes it into `SegmenterConfig` and into the ImageImputer constructor — `ImputerConfig` itself is never read by any component.

**Verdict: `ImputerConfig` should be removed.** It was a useful transitional design when one central config flowed to all components, but with typed per-component configs and `SpatialLayout` already carrying the same metadata, it adds indirection without value.

Removing it means:
- `ImputerFactory/data.py` loses one dataclass (`ImputerConfig`).
- `SegmenterConfig` gains model-metadata fields (populated by the Factory, defaulting to 0/empty).
- `ImageImputer.__init__` reads metadata from `SpatialLayout` instead of a config object, and accepts `use_amp` as a direct parameter.
- The Factory no longer constructs an intermediate `ImputerConfig` object.

### Segmenter constructor changes

**Before:** `Segmenter(config: ImputerConfig)` — reads everything from one object, including `config.segmenter_kwargs` as a weak dict.

**After:** `Segmenter(config: SegmenterConfig)` — the single `SegmenterConfig` carries both model metadata and strategy-specific params. The segmenter never sees `ImputerConfig`.

Example concrete signatures:

```python
class PatchSegmenter(BaseSegmenter):
    def __init__(self, config: SegmenterConfig):
        # reads: config.image_size, config.patch_size, config.grid_size, ...

class SLICSegmenter(BaseSegmenter):
    def __init__(self, config: SegmenterConfig, image_array):
        # reads: config.image_size, config.slic.n_segments, config.slic.compactness, ...
        # receives: image_array (raw PIL/numpy — a runtime dependency, not config)

class GradientGuidedSegmenter(BaseSegmenter):
    def __init__(self, config: SegmenterConfig, model, processor, image, text):
        # reads: config.image_size, config.gradient_guided.n_segments, ...
        # receives: model/processor/image/text (runtime deps for forward+backward pass)
```

**Key difference from the current design:** `image_array`, `model`, `processor`, `image`, `text` are no longer hidden inside a dict. They are explicit constructor arguments, surfaced via the Factory's per-strategy dispatch (see "Factory flow" below). The segmenter's constructor signature IS the documentation for what it needs.

### Masker constructor changes

**Before:** `CrossModalMeanMasker()` — no arguments.

**After:** `CrossModalMeanMasker(config: MaskerConfig)` — the single `MaskerConfig` carries strategy (and, in the future, per-strategy params for `AttentionMasker`).

```python
class CrossModalMeanMasker(BaseMasker):
    def __init__(self, config: MaskerConfig):
        # config.strategy → "crossmodal_mean"
        # config.crossmodal_mean → CrossModalMeanParams (currently empty)
        self._vision_masker = VisionMeanMasker(config=config)
        self._text_masker = TextAttentionMasker(config=config)

# Future:
# class AttentionMasker(BaseMasker):
#     def __init__(self, config: MaskerConfig):
#         # config.attention.temperature, config.attention.layer_name, ...
```

With typed per-strategy params, adding a new masker only requires:
1. A params dataclass (e.g. `AttentionParams`).
2. A new field on `MaskerConfig` with `default_factory`.
3. Adding the strategy literal to `MaskerConfig.strategy`.
No changes to the Factory or other maskers are needed.

### Factory flow after change

```
build(model, processor, image, text, segmenter_config=..., masker_config=..., use_amp=...)
  │
  ├─ 1. Infer model type (clip/siglip/siglip2)
  ├─ 2. Extract model dimensions (image_size, patch_size, n_channels)
  ├─ 3. Preprocess once → n_players_text, text_total_length
  ├─ 4. Enrich segmenter_config with model metadata:
  │      config.image_size = image_size
  │      config.patch_size = patch_size
  │      config.n_channels = n_channels
  │      config.model_type = model_type
  │      config.grid_size = grid_size
  │      config.n_players_image = provisional  (grid_size² for ViT, 0 for CNN)
  │      config.n_players_text = n_players_text
  │      config.text_total_length = text_total_length
  ├─ 5. Resolve strategy: auto → "patch" for ViT, "slic" for CNN
  ├─ 6. Create segmenter via per-strategy dispatch:
  │      patch:            cls(config)
  │      slic:             cls(config, image_array=input_image)
  │      gradient_guided:  cls(config, model=model, processor=processor,
  │                             image=input_image, text=input_text)
  ├─ 7. Sync config.n_players_image from segmenter.get_layout() (SLIC actual count)
  ├─ 8. Create masker: cls(config=mask_config)
  └─ 9. Assemble ImageImputer(model, processor, segmenter, masker,
                               inputs_original, inputs_raw,
                               input_image, input_text, use_amp)
```

Key differences from today:
- No `ImputerConfig` is constructed. Model metadata is written directly into the user's `SegmenterConfig` (step 4).
- Per-strategy runtime dependencies (`image_array`, `model`, `processor`) are passed as explicit args in a dispatch block (step 6), not hidden in a dict.
- `ImageImputer` reads spatial metadata from `segmenter.get_layout()` instead of a config object; `use_amp` is a direct parameter.

### `ImageImputer` constructor changes

**Before:** receives `config: ImputerConfig`, reads `image_size`, `patch_size`, etc. from it.

**After:** receives no `ImputerConfig`. Reads spatial metadata from `SpatialLayout` (already produced by `segmenter.get_layout()`), which carries the same fields. `use_amp` becomes a direct parameter.

```python
class ImageImputer:
    def __init__(
        self,
        model, processor, segmenter, masker,
        inputs_original: ProcessorOutput,
        inputs_raw: dict,
        input_image,
        input_text: str,
        use_amp: bool = False,
    ):
        self.model = model
        self.processor = processor
        self.segmenter = segmenter
        self.masker = masker
        self.use_amp = use_amp
        self.inputs_original = inputs_original
        self.inputs_raw = inputs_raw
        self.input_image = input_image
        self.input_text = input_text

        # All spatial metadata from layout
        self.layout = segmenter.get_layout()
        self.image_size = self.layout.image_size
        self.patch_size = self.layout.patch_size
        self.n_channels = self.layout.n_channels
        self.grid_size = self.layout.grid_size
        self.model_type = self.layout.model_type
```

This eliminates the `config` attribute from ImageImputer entirely. The layout already is the source of truth for spatial metadata; duplicating it in a config object was redundant.

### Unchanged boundaries

- **`VisionLanguageGame`**: unchanged — never sees config objects.
- **`SpatialLayout` / `PhysicalMask` / `ProcessorOutput`**: unchanged.
- **Segmenter / Masker registries**: unchanged. The `@register_segmenter("name")` decorator still maps string names to classes; only the call site in the Factory changes.

### Migration path

```
Phase 1 (current state): 
  build() uses segmenter / segmenter_kwargs / masker.
  ImputerConfig carries model metadata + component selection.
  Segmenter constructors take (config: ImputerConfig).
  Masker constructors take no arguments.

Phase 2 (this plan):
  → Add SegmenterConfig / MaskerConfig / per-strategy params dataclasses
  → Add model-metadata fields to SegmenterConfig (populated by Factory)
  → Replace build() signature: remove segmenter/segmenter_kwargs/masker,
    add segmenter_config/masker_config
  → Update all call sites (~9 experiment files + 3 notebooks)
  → Rewrite factory flow: enrich SegmenterConfig with model metadata,
    dispatch per-strategy with explicit runtime deps, no ImputerConfig
  → Update segmenter constructors: (config: ImputerConfig) → (config: SegmenterConfig, ...runtime_deps)
  → Update masker constructors: () → (config: MaskerConfig)
  → Update ImageImputer: read metadata from layout, accept use_amp directly
  → Delete ImputerConfig from data.py

Phase 3 (future):
  Existing callers already migrated; no deprecation period needed.
  New segmenters (AdaptiveSegmenter, etc.) add a params dataclass + 
  one case in the Factory dispatch — no dict-based parameter passing.

No deprecation bridge is planned. The change is mechanical; mypy
catches any missed spots at compile time. ImputerConfig is removed,
not deprecated — it has no public callers outside the Factory.
```

### Tradeoff summary

| Dimension | Phase 1 (strings + dict) | Phase 2 (typed configs, no ImputerConfig) |
|---|---|---|
| **Type safety** | `segmenter_kwargs` has none | Every parameter is a typed field |
| **User burden** | Low — strings are concise | Slightly higher — import + instantiate config class |
| **IDE support** | None for dict keys | Autocomplete + type hints for every field |
| **Runtime deps** | Hidden in dict, discovered by reading constructor | Explicit in constructor signature; Factory dispatch wires them |
| **Self-documentation** | `help(SlicParams)` → nada | `help(SlicParams)` shows all params + defaults |
| **Extensibility** | New segmenter = more undocumented dict keys | New segmenter = new params dataclass + one dispatch branch |
| **Classes in data.py** | 4 (ImputerConfig, SpatialLayout, PhysicalMask, ProcessorOutput) | 8: SegmenterConfig, MaskerConfig, and 6 per-strategy params dataclasses (PatchParams, SlicParams, GradientGuidedParams, CrossModalMeanParams, VisionMeanParams, TextAttentionParams). ImputerConfig removed; SpatialLayout, PhysicalMask, ProcessorOutput unchanged. |
| **`ImageImputer` constructor** | 8 args (incl. config) | 8 args (use_amp replaces config) |