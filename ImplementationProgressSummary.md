# ImageImputer — Implementation Progress Summary

> Last updated: 2026-05-27

## Implementation Progress Summary

| Module | Component | Status | Notes |
|---|---|---|---|
| **Data Types** | `ImputerConfig` | ✅ Done | Shared read-only config: model metadata + segmenter + masker + segmenter_kwargs |
| | `SpatialLayout` | ✅ Done | Player↔pixel/token mapping metadata |
| | `PhysicalMask` | ✅ Done | Concrete masks: `image_binary_mask` (N,C,H,W) + `text_attention_mask` (N,L) |
| | `ProcessorOutput` | ✅ Done | Standardized HuggingFace inputs wrapper |
| **Segmenters** | `BaseSegmenter` | ✅ Done | Abstract: `get_layout()` + `generate_masks()` |
| | `PatchSegmenter` | ✅ Done | Rigid grid, supports CLIP/SigLIP/SigLIP2 text masking |
| | `SLICSegmenter` | ✅ Done | CNN perceptual superpixels via skimage SLIC; CPU index-map → GPU scatter |
| | `GradientGuidedSegmenter` | ✅ Done | Future exploration: gradient extraction + watershed layout |
| | `AdaptiveSegmenter` | 🔬 Future | Future exploration: coarse-to-fine subdivision logic |
| | `HybridSegmenter` | ❌ Out of scope | Not planned for current phase |
| **Maskers** | `BaseMasker` | ✅ Done | Abstract: `apply(ProcessorOutput, PhysicalMask)` |
| | `VisionMeanMasker` | ✅ Done | Pure image occlusion via multiplicative binary mask (out-of-place) |
| | `TextAttentionMasker` | ✅ Done | Pure text occlusion via attention_mask replacement |
| | `CrossModalMeanMasker` | ✅ Done | Composite Pattern: delegates to VisionMeanMasker + TextAttentionMasker |
| | `AttentionMasker` | ⏳ Stub | Needs negative-infinity self-attention injection |
| **Core** | `ImageImputer` | ✅ Done | `forward_1d` + `forward_crossmodal` with batching & device mgmt |
| **Factory** | `ImageImputerFactory` | ✅ Done | Auto-detect model type, assemble PatchSegmenter + CrossModalMeanMasker |
| **Adapters** | `TensorOps` / `TorchOps` | ⏳ Stub | Interface defined; PyTorch-only (JAX out of scope) |
| **Integration** | `VisionLanguageGame` | ✅ Done | Thin adapter: delegates to Imputer, ~75 lines |

### Legend
- ✅ Done — fully implemented and tested
- ⏳ Stub — skeleton exists, logic outstanding
- 🔬 Future — future exploration / research item
- ❌ Not started — not yet created
- ❌ Out of scope — not planned

---

## Team Assignment & Task Planning

> Team size: 4 engineers + 1 PM. Structured into two specialized squads.

### Task Status Legend
| Icon | Meaning |
|---|---|
| ⬜ Not started | Task not yet begun |
| 🔄 Planning | Requirements being discussed, design in progress |
| 🔄 In progress | Work underway |
| ✅ Done | Completed and verified |
| ⬜ Waiting on A | Blocked until Team A provides input |

### Team Structure

```
┌─────────────────────────────────┐
│         PM (1 person)           │
│  Cross-team coordination        │
│  Requirements & prioritization  │
└──────────┬──────────────────────┘
           │
   ┌───────┴───────┐
   ▼               ▼
┌──────────┐  ┌──────────────┐
│ Team A   │  │  Team B      │
│ QA &     │  │  Feature Dev │
│ Adoption │  │  & Bug Fix   │
│ (2 ppl)  │  │  (2 ppl)     │
└──────────┘  └──────────────┘
```

---

### Team A — QA & Model Adoption (2 people)

**Mission**: Ensure all experiments pass and the Imputer + Game pipeline works correctly across CLIP model variants. Report blockers to Team B.

#### A1. Experiment Migration & Validation

| # | Target | Details | Success Criteria | Status |
|---|---|---|---|---|
| A1.1 | `experiments/faithfulness.py` | Migrate to `Game.game_huggingface` API | Same faithfulness metrics as `src` baseline (±1e-4) | ✅ Done |
| A1.2 | `experiments/insertion_deletion.py` | Migrate to `Game.game_huggingface` API | Same AID curve as `src` baseline | ✅ Done |
| A1.3 | `experiments/insertion_deletion_siglip.py` | Migrate + verify SigLIP support | Correct model type detection, no crash | 🔄 In progress |
| A1.4 | `experiments/pointing_game_banzhaf.py` | Migrate to `Game` API | Same PGR accuracy | ✅ Done |
| A1.5 | `experiments/pointing_game_shapley.py` | Migrate to `Game` API | Same PGR accuracy | ✅ Done |
| A1.6 | `experiments/pointing_game_crossmodal.py` | Migrate to `Game` API | Same PGR accuracy | ✅ Done |
| A1.7 | `experiments/explain_mscoco.py` | Migrate to `Game` API | Same top-k interaction overlap | 🔄 In progress |
| A1.8 | `experiments/explain_mscoco_siglip.py` | Migrate + verify SigLIP2 support | SigLIP2 model loads and runs | ⬜ Not started |

> **Notes:**
> - **A1.1** (`migrated/faithfulness.py`): HuggingFace model hardcoded to device 0 and OpenAI CLIP to device 1 (was the same logic in the inital pipelien) — two models run simultaneously on different GPUs to avoid OOM. When file moved back to experiements/ change PROJECT_ROOT reference from parent[2] to parent[1].
> - **A1.2** (`migrated/insertion_deletion.py`): When file moved back to experiements/ change PROJECT_ROOT reference from parent[2] to parent[1].

#### A2. Numerical Equivalence Regression

| # | Task | Details | Status |
|---|---|---|---|
| A2.1 | Build comparison harness | Script that runs same coalitions through `src` Game and `Game` Game, diffing outputs | ⬜ Not started |
| A2.2 | Snapshot baseline | Save reference outputs from all 8 experiments using `src` path | ⬜ Not started |
| A2.3 | CI-style gate | Exit code ≠ 0 if any experiment deviates > 1e-4 from baseline | ⬜ Not started |

#### A3. Cross-Model Adoption Tests

| # | Task | Details | Status |
|---|---|---|---|
| A3.1 | CLIP ViT-B/32 | Already validated in `example.ipynb` | ✅ Done |
| A3.2 | CLIP ViT-B/16 | Test with 196 image players (14×14 grid) | ⬜ Not started |
| A3.3 | CLIP ViT-L/14 | Test with 256 image players (16×16 grid), verify memory usage | ⬜ Not started |
| A3.4 | SigLIP base-patch16 | Test model_type detection + text masking logic | ⬜ Not started |
| A3.5 | SigLIP2 so400m | Test model_type detection (`siglip2` path) | ⬜ Not started |

#### A4. Feedback Loop to Team B

- File bug reports with minimal reproduction scripts
- Flag API rough edges (e.g., `inputs` / `processor` delegation pattern)
- Report performance regressions vs `src` baseline

---

### Team B — Feature Development & Bug Fix (2 people)

**Mission**: Implement CLIP-compatible segmenters, fix bugs reported by Team A, and optimize the Imputer pipeline.

#### B1. Bug Fix (Responsive — from Team A reports)

| # | Category | Expected Source | Status |
|---|---|---|---|
| B1.1 | Device placement | CPU/CUDA mismatch in edge cases | ⬜ Waiting on A |
| B1.2 | Crossmodal batch size | txt_bs ≠ img_bs correctness | ⬜ Waiting on A |
| B1.3 | Model type detection | Borderline model name patterns | ⬜ Waiting on A |
| B1.4 | Memory / OOM | Large models (ViT-L) with high budget | ⬜ Waiting on A |

#### B2. Segmenters (ordered by priority)

| # | Feature | Details | Priority | Status |
|---|---|---|---|---|
| B2.1 | `SLICSegmenter` | CPU: skimage SLIC → 2D index map. GPU: scatter coalition bits via index map | **High** | ✅ Done
| B2.2 | `GradientGuidedSegmenter` | Future exploration: gradient map → skimage watershed → non-uniform static layout | 🔬 Future | ⬜ Not started |
| B2.3 | `AdaptiveSegmenter` | Future exploration: coarse grid → score-driven subdivision → feedback loop. Requires `is_stateful=True` protocol | 🔬 Future | ⬜ Not started |

#### B2a. SLICSegmenter — CLIP-ResNet Validation

After B2.1 is complete, verify the SLICSegmenter + VisionMeanMasker pipeline works on CNN-based CLIP variants. Unlike ViT models where patches are rigid grids, CNN backbones process the full spatial input — SLIC superpixels are required to avoid OOD artifacts.

| # | Model | Backbone | Key Check | Status |
|---|---|---|---|---|
| B2a.1 | `openai/clip-rn50` | ResNet-50 | Correct model detection (should still return `"clip"`), SLIC layout produced, no crash | ⬜ Not started |
| B2a.2 | `openai/clip-rn101` | ResNet-101 | Same as above, verify memory usage | ⬜ Not started |
| B2a.3 | `openai/clip-rn50x4` | ResNet-50×4 | Larger ResNet variant — validate throughput | ⬜ Not started |

**Integration check**: Run `example.ipynb` equivalent with CLIP-ResNet + `segmenter="slic"`. Expected: AID values within ±5% of ViT-based results (SLIC superpixels may yield different but valid attributions). If the workflow fails (crash / NaN / OOM), B1 fixes take priority.

#### B3. Masker Extension

| # | Feature | Details | Status |
|---|---|---|---|
| B3.1 | `AttentionMasker` implementation | Hook self-attention, inject -inf mask matrices. Requires PyTorch `register_forward_hook` or HF `output_attentions` override | ⬜ Not started |

#### B4. Backend Adapter Extraction (PyTorch only)

| # | Feature | Details | Status |
|---|---|---|---|
| B4.1 | `TorchOps` extraction | Move inline PyTorch ops from Imputer/Segmenter into adapter | ⬜ Not started |

#### B6. Evaluation Infrastructure — Extract Reusable Libraries

> **B6 does NOT run independent experiments.** Its job is to extract shared evaluation logic (AID curve computation, faithfulness metrics, plotting) from **already-migrated** Team A experiments into reusable libraries. Team A owns the experiment scripts; B6 owns the tooling those scripts import.
>
> **Boundary rule**: If a change touches `experiments/*.py`, it belongs to Team A (migration). If it touches a new shared lib imported by multiple experiments, it belongs to B6.

| # | Task | Details | Blocked by | Status |
|---|---|---|---|---|
| B6.1 | AID curve library | Extract reusable AID computation + plotting from migrated A1.2 (`insertion_deletion.py`) and A1.7 (`explain_mscoco.py`). Must NOT re-implement experiment logic — only factor out common code | A1.2 ✅ / A1.7 ⬜ | 🔄 Planning |
| B6.2 | Faithfulness evaluation library | Extract reusable faithfulness metrics + harness from migrated A1.1 (`faithfulness.py`). Compare metrics before/after migration | A1.1 ✅ | ⬜ Not started |

| # | Task | Details | Status |
|---|---|---|---|
| B5.1 | `_repeat_inputs` memory | Replace `.expand().clone()` with stride tricks |✅ Done (Needs profiling validation)
| B5.2 | AMP support | `torch.autocast` for mixed-precision forward passes | ✅ Done (Opt-in, needs numeric validation)

---

### PM — Coordination & Oversight (1 person)

| # | Responsibility | Status |
|---|---|---|
| P1 | Maintain this document as the single source of truth | 🔄 Ongoing |
| P2 | Weekly sync: Team A reports blockers → Team B prioritizes fixes | 🔄 Ongoing |
| P3 | Triage A4 bug reports, assign severity, track resolution | ⬜ Pending |
| P4 | Review API decisions (naming, data formats, public surface) | 🔄 Ongoing |
| P5 | Sign off on experiment migration checkpoints (A1.1–A1.8) | ⬜ Pending |
| P6 | Maintain comparison harness (A2.1) as gatekeeper for merges | ⬜ Pending |

---

### Task Dependencies

```
Team A                              Team B
──────                              ──────
A1.1–A1.8 (migrate experiments)    B2.1 SLICSegmenter
    │                                   │
    ├─ A2 (equivalence tests) ──────────┤ (bug reports)
    │       │                           │
    │       ▼                           ▼
    ├─ A3 (cross-model) ──────────► B1 (bug fixes)
    │       │                           │
    │       ▼                           ▼
    └─ A4 (feedback) ─────────────► B2a CLIP-ResNet validation
                                        │
                                        ▼
                                    B3, B5 (extensions)
                                    B2.2–B2.3 (future)
                                        │
                    ┌───────────────────┘
                    ▼
            B6 (extract libs from A1.1, A1.2, A1.7)
```

| Dependency | Blocker | Blocked by |
|---|---|---|
| B1 (bug fix) | A2/A3 reports | Team A findings |
| B2a (ResNet validation) | B2.1 (SLIC) complete | B2.1 |
| B2.2–B2.3 (GGS/AS) | N/A (future exploration) | Future milestone |
| A3 (cross-model) | A1 completion | All experiments pass |
| B6.1 (AID curve lib) | A1.2 + A1.7 migrated | Team A migration |
| B6.2 (faithfulness lib) | A1.1 migrated | Team A migration |

---

### Milestone Schedule

| Week | Team A | Team B | PM Gate |
|---|---|---|---|
| W1 | A1.1–A1.4, A2.1 | B1 (bug fixes), B4.1 (TorchOps) | Experiments 1–4 pass |
| W2 | A1.5–A1.8, A2.2–A2.3 | B2.1 (SLICSegmenter) | All 8 experiments pass |
| W3 | A3.1–A3.5 (cross-model) | B2a (CLIP-ResNet validation) | SLIC + ResNet workflow green |
| W4 | A4 (feedback loop) | B6.1–B6.2 (extract eval libs from A1.1/A1.2/A1.7) | Feature freeze, integration test |

---

### Known Issues (tracked for B1)

- **Crossmodal edge-case processor calls**: When `budget_image % batch_size ≠ 0` or `budget_text % batch_size ≠ 0`, the last image and/or text batch have incomplete sizes (e.g., `img_bs=15, txt_bs=51` for `batch_size=64, budget_image=4559, budget_text=115`). The 2 (img batches) × 2 (text batches) = 4 combinations yield 3 cases where `img_bs ≠ txt_bs`. In those cases `_preprocess_batch()` must re-invoke the HF processor to create inputs with matching batch dimensions. The original `src` code has the same behavior (it calls `processor_function` directly in the equivalent branches), so this is not a regression — it is inherent to the double-loop crossmodal design. Total extra calls per `forward_crossmodal`: at most 3 (~2 ms each, negligible).
