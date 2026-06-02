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

- `ImputerConfig`: shared read-only configuration (produced by Factory, consumed by all modules). Contains model metadata (`model_type`, `image_size`, `patch_size`, `n_channels`, `n_players_image`, `n_players_text`, `grid_size`, `text_total_length`, `use_amp`). Does NOT carry component selection or user-provided parameters — those live in `SegmenterConfig` / `MaskerConfig`. (Planned: remove `segmenter`, `masker`, `segmenter_kwargs` fields — see "Future API Evolution" section.)
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
| `SegmenterConfig` / `MaskerConfig` are user-provided; `ImputerConfig` is Factory-produced | Clear separation: user controls component selection + params, factory controls model metadata |

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

2. **ImputerConfig as shared truth**: `ImputerConfig` is created once by the Factory and shared read-only across Segmenter, Masker, and Imputer. This eliminates scattered model introspection. It contains only model metadata — component selection and parameters live in `SegmenterConfig` / `MaskerConfig` instead. (Planned: remove the interim `segmenter_kwargs` approach — see "Future API Evolution".)

3. **Imputer owns the inputs**: `ImageImputer` stores `inputs_original` (ProcessorOutput), `inputs_raw` (HF dict for `.tokens()`), and `input_image`/`input_text` (for crossmodal edge cases where batch sizes diverge).

4. **Game is a thin shell**: `VisionLanguageGame` delegates all masking/batching/model-forward to the Imputer. It only handles shapiq scheduling (normalization values, player counts).

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

1. **`segmenter_kwargs` is a type hole.** IDE autocomplete cannot suggest keys, mypy cannot catch typos (`n_segments` vs `n_segements`), and per-strategy parameters mix in one flat dict (`image_array` for SLIC coexists with `model` for gradient-guided).

2. **Mixed concerns in `ImputerConfig`.** Today it carries model metadata (`image_size`, `model_type`, etc.) **and** component selection (`segmenter`, `masker`) **and** a dict of strategy-specific params. Design doc says "ImputerConfig is produced by Factory" — but `segmenter_kwargs` is user-provided, violating that boundary.

3. **No migration path for new parameters.** Every new segmenter adds more undocumented keys to the same `dict`. Callers must grep the segmenter constructor to learn what keys are accepted.

### Proposed solution: `SegmenterConfig` + `MaskerConfig`

Add two new user-facing configuration dataclasses to `ImputerFactory/data.py`. `build()` accepts these instead of strings; the factory uses them to select and parameterize components. `ImputerConfig` sheds its component-selection fields and becomes a pure model-metadata container.

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
    User-facing configuration for spatial division.

    strategy="auto" delegates selection to the Factory
    (ViT → "patch", CNN → "slic") — same as today's segmenter=None.
    """
    strategy: Literal["auto", "patch", "slic", "gradient_guided"] = "auto"
    patch: PatchParams = field(default_factory=PatchParams)
    slic: SlicParams = field(default_factory=SlicParams)
    gradient_guided: GradientGuidedParams = field(default_factory=GradientGuidedParams)

    @property
    def active_params(self):
        """Return the params dataclass for the active strategy."""
        return getattr(self, self.strategy, None)


@dataclass
class MaskerConfig:
    """
    User-facing configuration for occlusion strategy.

    Currently maskers take no runtime parameters, so this acts
    as a simple selector. Future maskers (e.g. AttentionMasker)
    will add their own params fields here.
    """
    strategy: Literal["auto", "crossmodal_mean", "vision", "text"] = "auto"
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

### `ImputerConfig` simplification

Remove three fields from `ImputerConfig`:

| Remove | Replaced by |
|---|---|
| `segmenter: Optional[str]` | `SegmenterConfig.strategy` |
| `masker: Optional[str]` | `MaskerConfig.strategy` |
| `segmenter_kwargs: dict` | Typed fields in per-strategy params (e.g. `SlicParams.n_segments`) |

`ImputerConfig` retains: `model_type`, `image_size`, `patch_size`, `n_channels`, `n_players_image`, `n_players_text`, `grid_size`, `text_total_length`, `use_amp`.

This restores the original contract: **ImputerConfig is produced by the Factory (model metadata only)**. Component selection becomes an entirely separate axis delivered by the caller.

### Segmenter constructor changes

**Before:** `Segmenter(config: ImputerConfig)` — reads everything from one object, including `config.segmenter_kwargs` as a weak dict.

**After:** `Segmenter(config: ImputerConfig, seg_config: SegmenterConfig)` — receives model metadata (config) and its own typed parameters (seg_config) as two separate arguments.

**Runtime dependency injection changes.** Today, the factory injects per-segmenter runtime objects (image_array for SLIC, model/processor for gradient-guided) by mutating `config.segmenter_kwargs`:

```python
config.segmenter_kwargs.setdefault("image_array", input_image)
config.segmenter_kwargs.setdefault("model", model)
config.segmenter_kwargs.setdefault("processor", processor)
```

After the change, these become explicit constructor arguments, surfaced via per-strategy dispatch in `_create_segmenter`:

```python
def _create_segmenter(
    self,
    config: ImputerConfig,
    seg_config: SegmenterConfig,
    model, processor, input_image, input_text,
) -> BaseSegmenter:
    cls = get_segmenter(seg_config.strategy)

    if seg_config.strategy == "slic":
        return cls(config=config, seg_config=seg_config, image_array=input_image)

    if seg_config.strategy == "gradient_guided":
        return cls(config=config, seg_config=seg_config,
                   model=model, processor=processor,
                   image=input_image, text=input_text)

    # "patch" / default — no runtime deps beyond config + seg_config
    return cls(config=config, seg_config=seg_config)
```

This makes the contract explicit: `SLICSegmenter` requires `image_array`, `GradientGuidedSegmenter` requires `model` + `processor` + `image` + `text`. A new segmenter simply adds its own case here; the parameter names in the constructor become the documentation.

### Masker constructor changes

Currently maskers take no arguments (`CrossModalMeanMasker.__init__(self)`). After the change, all maskers accept `mask_config: MaskerConfig` to keep the constructor signature uniform, even if the current maskers ignore it:

```python
class CrossModalMeanMasker(BaseMasker):
    def __init__(self, mask_config: MaskerConfig):
        self._vision_masker = VisionMeanMasker(mask_config=mask_config)
        self._text_masker = TextAttentionMasker(mask_config=mask_config)
```

This ensures `AttentionMasker(mask_config)` can be added later without another factory rewrite.

### Factory flow after change

```
build(model, processor, image, text, segmenter_config=..., masker_config=..., use_amp=...)
  │
  ├─ 1. Infer model type (clip/siglip/siglip2)
  ├─ 2. Extract model dimensions (image_size, patch_size, n_channels)
  ├─ 3. Preprocess once → n_players_text, text_total_length
  ├─ 4. Build ImputerConfig (model metadata only — no segmenter/masker fields)
  ├─ 5. Resolve segmenter_config.strategy (auto → "patch" for ViT, "slic" for CNN)
  ├─ 6. _create_segmenter(config, seg_config, model, processor, image, text)
  └─ 7. _create_masker(mask_config)
```

Steps 1–4 are unchanged from today. The key differences:
- Step 4 no longer writes `segmenter`/`masker`/`segmenter_kwargs` into `ImputerConfig`.
- Step 6 passes per-strategy runtime deps explicitly rather than hiding them in a dict.
- Step 7 passes `mask_config` instead of relying on parameterless constructors.

### No changes outside the Factory boundary

- **`ImageImputer`**: unchanged. Its constructor signature and runtime behavior are unaffected; `ImputerConfig` simply has fewer fields.
- **`VisionLanguageGame`**: unchanged — never sees config objects.
- **`SpatialLayout` / `PhysicalMask` / `ProcessorOutput`**: unchanged.
- **Segmenter / Masker registries**: unchanged. The `@register_segmenter("name")` decorator still maps string names to classes; only the call site in `_create_segmenter` changes.

### Migration path

```
Phase 1 (current state): 
  build() uses segmenter / segmenter_kwargs / masker.
  ImputerConfig carries model metadata + component selection.

Phase 2 (this plan):
  → Add SegmenterConfig / MaskerConfig / per-strategy params dataclasses
  → Replace build() signature: remove three old params, add two new ones
  → Update all call sites (~9 experiment files + 3 notebooks)
  → Strip segmenter / masker / segmenter_kwargs from ImputerConfig
  → Update segmenter constructors: (config) → (config, seg_config)
  → Update masker constructors: () → (mask_config)
  → Remove per-strategy segmenter_kwargs.setdefault() from factory
  → Add explicit per-strategy dispatch in _create_segmenter

Phase 3 (future):
  Existing callers already migrated; no deprecation period needed.
  New segmenters (AdaptiveSegmenter, etc.) add a params dataclass + 
  one case in _create_segmenter dispatch — no dict-based parameter passing.

No deprecation bridge is planned. The change is mechanical and affects
~12 known call sites; a sed / regex pass covers the migration in minutes.
Type errors from mypy catch any missed spots at compile time.
```

### Tradeoff summary

| Dimension | Phase 1 (strings + dict) | Phase 2 (typed configs) |
|---|---|---|
| **Type safety** | `segmenter_kwargs` has none | Every parameter is a typed field |
| **User burden** | Low — strings are concise | Slightly higher — import + instantiate config class |
| **IDE support** | None for dict keys | Autocomplete + type hints for every field |
| **Runtime deps** | Hidden in dict, discovered by reading constructor | Explicit in `_create_segmenter` dispatch + constructor signature |
| **Self-documentation** | `help(SlicParams.__init__)` vs grepping source | `help(SlicParams)` shows all params + defaults |
| **Extensibility** | New segmenter = more undocumented dict keys | New segmenter = new params dataclass + one dispatch branch |
| **Lines of config code** | 3 stub classes | ~6–8 small classes (~60 lines total) |

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