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
│         grid_size=7, text_total_length=10, segmenter=None,
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
