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
- **Examples**: `PatchSegmenter` (rigid grid), `GradientGuidedSegmenter` (gradient-based), `AdaptiveSegmenter` (score-driven)

### Masker
**Sole responsibility**: Feature occlusion — apply the `PhysicalMask` to `ProcessorOutput`.

- **Owns**: the occlusion strategy (multiplicative, attention injection, etc.)
- **Does NOT own**: mask generation, model forward pass, batching logic
- **Contract**:
  - `apply(processor_output, physical_mask)` → returns `ProcessorOutput` (modified)
- **Must NOT**: access `model`, call `processor`, generate masks from coalitions
- **Must**: clone inputs before mutation (never mutate the original)
- **Examples**: `CrossModalMeanMasker` (pixel multiply + attention swap), `AttentionMasker` (negative-infinity injection)

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
  - `build(model, processor, input_image, input_text, accelerator=...)` → `ImageImputer`
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

- `ImputerConfig`: shared read-only configuration (produced by Factory, consumed by all modules). Contains model metadata, accelerator selection, and `segmenter_kwargs` for variable block sizing.
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

## End-to-End Workflow (from `example.ipynb`)

Using CLIP ViT-B/32 explaining "black dog next to a yellow hydrant" as an example, tracing the full data flow from raw input to final results.

### Phase 0: Environment Setup

```python
# ── Step 0.1: Load model ─────────────────────────────────────────────
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
model.to('cuda')
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# ── Step 0.2: Load data ──────────────────────────────────────────────
input_text = "black dog next to a yellow hydrant"   # 8 tokens before BOS/EOS
input_image = Image.open("assets/dog_and_hydrant.png")  # 224×224 RGB
```

No Imputer involved yet — just loading the HuggingFace model and a PIL image.

---

### Phase 1: Factory Assembles the Imputer

```python
factory = ImageImputerFactory()
imputer = factory.build(model, processor, input_image, input_text)
```

**Internal process:**

```
factory.build(model, processor, input_image, input_text)
│
├─ 1. _infer_model_type(model)
│     └─ model.name_or_path → "openai/clip-vit-base-patch32" → model_type = "clip"
│
├─ 2. Extract model dimensions
│     └─ image_size=224, patch_size=32, n_channels=3
│
├─ 3. _preprocess(processor, image, text, "clip")
│     └─ processor(images=img, text=txt, return_tensors="pt", padding=True)
│     └─ Output: {"pixel_values": (1,3,224,224), "input_ids": (1,10), "attention_mask": (1,10)}
│
├─ 4. _count_text_players(inputs_dict, "clip")
│     └─ CLIP: input_ids.size(1) - 2 = 10 - 2 = 8
│     └─ n_players_text = 8, text_total_length = 10
│
├─ 5. Build ImputerConfig
│     └─ ImputerConfig(model_type="clip", image_size=224, patch_size=32,
│         n_channels=3, n_players_image=49, n_players_text=8,
│         grid_size=7, text_total_length=10, accelerator=None,
│         segmenter_kwargs={})
│
├─ 6. _create_segmenter(config)
│     └─ PatchSegmenter(config)  ← receives shared config
│     └─ Produces SpatialLayout(n_players_image=49, n_players_text=8, ...)
│
├─ 7. _create_masker(...)
│     └─ CrossModalMeanMasker()
│
├─ 8. Build ProcessorOutput + assemble ImageImputer
│     └─ inputs_original: ProcessorOutput(pixel_values, input_ids, attention_mask)
│     └─ inputs_raw: raw HF dict (preserves .tokens() etc.)
│     └─ config passed to ImageImputer  ← shared across all components
│     └─ input_image, input_text stored on imputer (for crossmodal edge cases)
│
└─ Returns: ImageImputer(model, processor, segmenter, masker, inputs_original, ...)
```

**Assembly result:**
| Component | Instance | Key parameters |
|---|---|---|
| Segmenter | `PatchSegmenter` | grid=7×7, n_players_image=49 |
| Masker | `CrossModalMeanMasker` | Image multiplicative mask + text attention swap |
| Layout | `SpatialLayout` | n_players=57 (49 img + 8 txt) |
| Inputs | `ProcessorOutput` | 1-sample preprocessed result |

---

### Phase 2: Game Wrapper

```python
game = VisionLanguageGame(imputer, batch_size=64)
```

**Internal process:**

```
VisionLanguageGame.__init__(imputer, batch_size=64)
│
├─ self.n_players_image = imputer.n_players_image  # 49
├─ self.n_players_text  = imputer.n_players_text   # 8
│
├─ Compute normalization baselines:
│   coalitions = [[False×57], [True×57]]  # empty coalition + full coalition
│   game_output = self.value_function(coalitions)
│     └─ Internally calls imputer.forward_1d(coalitions, batch_size=64)
│     └─ Returns [empty_value, full_value] = [-0.92, 27.3]
│
├─ self.empty_value = -0.92
├─ self.full_value  = 27.3
│
└─ super().__init__(n_players=57, normalize=True, normalization_value=-0.92)
    └─ shapiq.Game subtracts -0.92 from every value_function return
```

**Delegate properties provided by Game:**
- `game.inputs` → `imputer.inputs_raw`（HF BatchEncoding, supports `.tokens()`）
- `game.processor` → `imputer.processor`（for denormalization）

---

### Phase 3: FIxLIP Initialization

```python
fixlip = FIxLIP(n_players_image=49, n_players_text=8, max_order=2,
                p=0.5, mode="banzhaf", random_state=0)
```

**Internal process:**

```
FIxLIP.__init__(n_players_image=49, n_players_text=8, max_order=2, p=0.5)
│
├─ mode="banzhaf" → use weighted Banzhaf sampling
├─ max_order=2 → compute 1st-order attributions + 2nd-order interactions
├─ is_crossmodal=True
│
├─ Create sampler_image: CoalitionSampler(n_players=49, ...)
│   └─ sampling_weights[k] = C(49,k) * p^k * (1-p)^(49-k)
│
├─ Create sampler_text: CoalitionSampler(n_players=8, ...)
│   └─ sampling_weights[k] = C(8,k) * p^k * (1-p)^(8-k)
│
└─ Total players: n_players = 49 + 8 = 57
    Number of interactions = C(57,0) + C(57,1) + C(57,2) = 1 + 57 + 1596 = 1654
    Of which 1596 are 2nd-order interactions
```

---

### Phase 4: Compute Explanation

```python
interaction_values = fixlip.approximate_crossmodal(game, budget=2**19)
```

This is the core and most time-consuming step of the entire pipeline.

**4.1 Budget split（`split_budget`）**

```
budget = 2^19 = 524,288
├─ n_players_text=8 < n_players_image=49
├─ budget_text = √524288 × 8/49 ≈ 115
├─ budget_image = 524288 / 115 ≈ 4559
└─ Final: 4559 image coalitions × 115 text coalitions
```

**4.2 Sample coalitions**

```
sampler_image.sample(4559)  → coalitions_image: (4559, 49) bool
sampler_text.sample(115)    → coalitions_text:  (115, 8)   bool
```

Each coalition is a 0/1 vector indicating which patches/tokens are present (1) or absent (0).

**4.3 Evaluate coalition values（`game.value_function_crossmodal`）**

```
coalitions_image (4559, 49)  ─┐
                               ├─► imputer.forward_crossmodal()
coalitions_text  (115, 8)    ─┘       │
                                       ▼
    ① Segmenter.generate_masks(coalitions_image, coalitions_text)
       └─ PhysicalMask:
          image_binary_mask: (4559, 3, 224, 224)  # patch-level → pixel-level expansion
          text_attention_mask: (115, 10)           # BOS/EOS auto-padded

    ② Double-loop batch processing (batch_size=64):
       outer: 4559/64 ≈ 72 image batches
         inner: 115/64 ≈ 2 text batches
           ├─ Masker.apply() → pixel_values *= image_mask, attention_mask ← text_mask
           ├─ model(**masked_inputs) → logits_per_image
           └─ Collect logits matrix

    ③ Return: (4559, 115) np.array  ← 4559×115 similarity scores
```

**Batch processing illustration:**

```
              text coalition 1  text coalition 2  ...  text coalition 115
img coal 1    logit(1,1)        logit(1,2)             logit(1,115)
img coal 2    logit(2,1)        logit(2,2)             logit(2,115)
   ...
img coal 4559 logit(4559,1)     logit(4559,2)          logit(4559,115)
```

Each cell = model similarity under one (image occlusion, text occlusion) combination.

**4.4 Normalize + reshape**

```python
coalition_values_crossmodal = (4559, 115) - game.normalization_value  # subtract -0.92
# Reshape to (4559×115,) = (524285,) 1D vector
coalition_values = coalition_values_crossmodal.reshape(-1)

# Align image and text coalitions via repeat/tile
coalitions_matrix = (524285, 57)  # 49 image bits + 8 text bits
├─ First 49 cols: sampler_image coalitions, each repeated 115 times
└─ Last 8 cols:   sampler_text coalitions, tiled 4559 times
```

**4.5 Weighted regression → interaction coefficients**

```
aggregate(coalition_matrix (524285, 57), coalition_values (524285,), regression_weights)
│
├─ interaction_lookup: {(): idx0, (0,): idx1, ..., (48,55): idx1653}
│   └─ 1654 interactions, of which 1596 are 2nd-order
│
├─ Build regression matrix X (524285, 1654):
│   X[:, i] = 1 iff all players of interaction i are present in the coalition
│   e.g., interaction=(33,55) → coalition[33] × coalition[55]
│
├─ Weighted least squares regression:
│   solve_regression(X, y, weights)
│   └─ Solves for φ[1654] ← contribution coefficient per interaction
│
└─ Returns: InteractionValues(n_players=57, max_order=2, index="FWBII")
```

**Regression principle:** coalition_value ≈ Σ φ_interaction × 𝟙[interaction ⊆ coalition]

The model's similarity score is decomposed into the sum of all interaction contributions.

---

### Phase 5: Results Output

```python
print(interaction_values)
```

```
InteractionValues(
    index=FWBII, max_order=2, n_players=57,
    baseline_value=-0.9236,               # empty coalition value
    Top 10 interactions:
        (54,):     2.24   ← 1st-order: token "black" (player 54)
        (33, 55):  1.57   ← 2nd-order: synergy of patch 33 × token "dog"
        (28, 50):  1.35   ← 2nd-order: synergy of patch 28 × token "yellow"
        (29, 50):  1.31   ← 2nd-order: same
        (36, 50):  1.26   ← 2nd-order: same
        (55, 56):  1.24   ← 2nd-order: intra-text interaction "dog" × "next"
        (52,):    -1.39   ← 1st-order: token 52 has negative contribution
        (56,):    -1.76   ← 1st-order
        (50,):    -4.61   ← 1st-order: token "yellow" strong negative
        (55,):    -5.68   ← 1st-order: token "dog" strong negative
)
```

**Interpretation:**
- **(54,): +2.24** — the token "black" alone contributes +2.24 to the similarity score
- **(33, 55): +1.57** — when patch 33 and "dog" are both present, they produce an extra +1.57 synergy beyond their individual contributions
- **(50,): -4.61** — "yellow" alone actually reduces similarity (likely because the image contains a yellow hydrant, but the word "yellow" needs other tokens to correctly localize)

---

### Phase 6: Visualization

```python
src.plot.plot_image_and_text_together(
    img=input_image_denormalized,
    text=text_tokens,           # ['black', 'dog', 'next', 'to', 'a', 'yellow', 'hydrant', '']
    image_players=list(range(49)),
    iv=interaction_values,
    plot_interactions=True,     # draw 2nd-order interaction lines
    top_k=14,                   # show top-14 interactions
    normalize_jointly=True,     # jointly normalize 1st + 2nd order
)
```

**Visualization elements:**
- **Image-side heatmap**: 1st-order attribution (individual contribution per patch)
- **Text-side bar chart**: 1st-order attribution (individual contribution per token)
- **Cross-modal lines**: 2nd-order interaction（colored lines between patch ↔ token, width/color = interaction strength）
- **Intra-text lines**: 2nd-order interaction（token ↔ token interactions）

---

### Complete Pipeline Diagram

```
  Input Layer        Assembly Layer              Computation Layer          Output Layer
┌────────┐    ┌──────────────────┐    ┌─────────────────────┐    ┌──────────────┐
│ Image  │    │ ImageImputer     │    │ FIxLIP              │    │ (54,): 2.24  │
│ Text   │───►│ Factory.build()  │    │ .approximate_       │    │ (33,55):1.57 │
│ Model  │    │                  │    │  crossmodal(game,   │    │ (28,50):1.35 │
│Proc.   │    │ ┌PatchSegmenter┐ │    │  budget=2^19)       │    │    ...       │
└────────┘    │ │CrossModal    │ │    │                     │    └──────────────┘
              │ │MeanMasker    │ │    │ ① split_budget     │
              │ └──────────────┘ │    │ ② sample coalitions │
              │         │        │    │ ③ game.value_func_  │
              │         ▼        │    │   crossmodal()      │
              │ ┌ImageImputer┐  │    │   └→ imputer.fwd_   │    ┌──────────────┐
              │ │forward_1d() │  │    │      crossmodal()   │    │  Heatmap +   │
              │ │fwd_cross()  │◄─┼────┤ ④ aggregate(WLS)   │───►│  Interaction │
              │ └────────────┘  │    └─────────────────────┘    │  Lines       │
              └──────────────────┘                              └──────────────┘
```

---

## Key Design Decisions

1. **"CPU Planning, GPU Execution"**: Segmenters produce integer index maps once on CPU (via skimage). Thousands of coalition→mask translations happen purely on GPU via native tensor ops.

2. **ImputerConfig as shared truth**: `ImputerConfig` is created once by the Factory and shared read-only across Segmenter, Masker, and Imputer. This eliminates scattered model introspection. `segmenter_kwargs` carries variable block-size parameters from Factory → Segmenter.

3. **Imputer owns the inputs**: `ImageImputer` stores `inputs_original` (ProcessorOutput), `inputs_raw` (HF dict for `.tokens()`), and `input_image`/`input_text` (for crossmodal edge cases where batch sizes diverge).

4. **Game is a thin shell**: `VisionLanguageGame` delegates all masking/batching/model-forward to the Imputer. It only handles shapiq scheduling (normalization values, player counts).

---

## Current Implementation Details

### `ImputerFactory/data.py`
Four dataclasses serve as the universal data protocol:
- **`ImputerConfig`**: Read-only configuration produced by Factory. Contains `model_type`, `image_size`, `patch_size`, `n_channels`, `n_players_image`, `n_players_text`, `grid_size`, `text_total_length`, `accelerator`, and `segmenter_kwargs` (forwarded to Segmenter for variable block sizing). Shared across all components.
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

### `ImputerFactory/maskers/mean.py` — CrossModalMeanMasker
- Image: `pixel_values *= image_binary_mask` (zero-mean normalization → mean fill)
- Text: replaces `attention_mask` with coalition-derived mask
- Clones inputs to avoid mutation

### `ImputerFactory/core/imputer.py` — ImageImputer
- Constructor: receives `config: ImputerConfig` — model metadata derived from config, not from `model` directly
- **`forward_1d(coalitions, batch_size)`**: Splits coalitions → generates masks → batches → masks → model → extracts diagonal
- **`forward_crossmodal(coalitions_img, coalitions_txt, batch_size)`**: Double loop (image outer, text inner). Edge case: when txt_bs ≠ img_bs, re-processes via `_preprocess_batch()` using stored `input_image`/`input_text`
- **`_model_forward()`**: Auto-detects model device, moves inputs before forward
- Stores: `config`, `inputs_original`, `inputs_raw`, `input_image`, `input_text`, `model`, `processor`, `segmenter`, `masker`, `layout`

### `ImputerFactory/factory.py` — ImageImputerFactory
- `build(model, processor, input_image, input_text, accelerator=None)`:
  1. Infers model type (clip/siglip/siglip2)
  2. Extracts model dimensions and computes derived values (grid_size, n_players_image)
  3. Preprocesses once to determine `n_players_text` + `text_total_length`
  4. **Builds `ImputerConfig`** — single source of truth for all metadata, accelerator selection, and `segmenter_kwargs`
  5. Creates segmenter via `_create_segmenter(config)` — config flows into Segmenter
  6. Creates `CrossModalMeanMasker`
  7. Wires `ProcessorOutput` + raw dict + config + raw image/text into `ImageImputer`

### `Game/game_huggingface.py` — VisionLanguageGame
- Constructor: `VisionLanguageGame(imputer, batch_size=64, verbose=False)`
- `n_players_image` / `n_players_text` from imputer layout
- `inputs` / `processor` properties delegate to imputer (backward compat)
- `value_function()` → `imputer.forward_1d()`
- `value_function_crossmodal()` → `imputer.forward_crossmodal()`

---

## Known Issues

- **Crossmodal edge-case processor calls**: When `budget_image % batch_size ≠ 0` or `budget_text % batch_size ≠ 0`, the last image and/or text batch have incomplete sizes (e.g., `img_bs=15, txt_bs=51` for `batch_size=64, budget_image=4559, budget_text=115`). The 2 (img batches) × 2 (text batches) = 4 combinations yield 3 cases where `img_bs ≠ txt_bs`. In those cases `_preprocess_batch()` must re-invoke the HF processor to create inputs with matching batch dimensions. The original `src` code has the same behavior (it calls `processor_function` directly in the equivalent branches), so this is not a regression — it is inherent to the double-loop crossmodal design. Total extra calls per `forward_crossmodal`: at most 3 (~2 ms each, negligible).
- `_repeat_inputs` uses `.expand().clone()` which duplicates memory; could be optimized with stride tricks
- No mixed-precision (AMP) support yet — relevant for larger models