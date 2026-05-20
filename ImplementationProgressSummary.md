# ImageImputer — Implementation Progress Summary

> Last updated: 2026-05-13

## Implementation Progress Summary

| Module | Component | Status | Notes |
|---|---|---|---|
| **Data Types** | `ImputerConfig` | ✅ Done | Shared read-only config: model metadata + accelerator + segmenter_kwargs |
| | `SpatialLayout` | ✅ Done | Player↔pixel/token mapping metadata |
| | `PhysicalMask` | ✅ Done | Concrete masks: `image_binary_mask` (N,C,H,W) + `text_attention_mask` (N,L) |
| | `ProcessorOutput` | ✅ Done | Standardized HuggingFace inputs wrapper |
| **Segmenters** | `BaseSegmenter` | ✅ Done | Abstract: `get_layout()` + `generate_masks()` |
| | `PatchSegmenter` | ✅ Done | Rigid grid, supports CLIP/SigLIP/SigLIP2 text masking |
| | `SLICSegmenter` | 🔄 In progress | CNN perceptual superpixels via skimage SLIC; CPU index-map → GPU scatter |
| | `GradientGuidedSegmenter` | ⏳ Stub | Needs gradient extraction + watershed layout |
| | `AdaptiveSegmenter` | ⏳ Stub | Needs coarse-to-fine subdivision logic |
| | `HybridSegmenter` | ❌ Out of scope | Not planned for current phase |
| **Maskers** | `BaseMasker` | ✅ Done | Abstract: `apply(ProcessorOutput, PhysicalMask)` |
| | `VisionMeanMasker` | ✅ Done | Pure image occlusion via multiplicative binary mask (out-of-place) |
| | `TextAttentionMasker` | ✅ Done | Pure text occlusion via attention_mask replacement |
| | `CrossModalCompositeMasker` | ✅ Done | Composite Pattern: delegates to VisionMeanMasker + TextAttentionMasker |
| | `CrossModalMeanMasker` | ⚠️ Deprecated | Kept for backward compat; prefer CrossModalCompositeMasker |
| | `AttentionMasker` | ⏳ Stub | Needs negative-infinity self-attention injection |
| **Core** | `ImageImputer` | ✅ Done | `forward_1d` + `forward_crossmodal` with batching & device mgmt |
| **Factory** | `ImageImputerFactory` | ✅ Done | Auto-detect model type, assemble PatchSegmenter + CrossModalCompositeMasker |
| **Adapters** | `TensorOps` / `TorchOps` / `JaxOps` | ⏳ Stub | Interface defined, implementations pending |
| **Integration** | `VisionLanguageGame` | ✅ Done | Thin adapter: delegates to Imputer, ~75 lines |

### Legend
- ✅ Done — fully implemented and tested
- ⏳ Stub — skeleton exists, logic outstanding
- ❌ Not started — not yet created

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
| A1.3 | `experiments/insertion_deletion_siglip.py` | Migrate + verify SigLIP support | Correct model type detection, no crash | ⬜ Not started |
| A1.4 | `experiments/pointing_game_banzhaf.py` | Migrate to `Game` API | Same PGR accuracy | ⬜ Not started |
| A1.5 | `experiments/pointing_game_shapley.py` | Migrate to `Game` API | Same PGR accuracy | ✅ Done |
| A1.6 | `experiments/pointing_game_crossmodal.py` | Migrate to `Game` API | Same PGR accuracy | ✅ Done |
| A1.7 | `experiments/explain_mscoco.py` | Migrate to `Game` API | Same top-k interaction overlap | ⬜ Not started |
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

**Mission**: Implement CLIP-compatible accelerator segmenters, fix bugs reported by Team A, and optimize the Imputer pipeline.

#### B1. Bug Fix (Responsive — from Team A reports)

| # | Category | Expected Source | Status |
|---|---|---|---|
| B1.1 | Device placement | CPU/CUDA mismatch in edge cases | ⬜ Waiting on A |
| B1.2 | Crossmodal batch size | txt_bs ≠ img_bs correctness | ⬜ Waiting on A |
| B1.3 | Model type detection | Borderline model name patterns | ⬜ Waiting on A |
| B1.4 | Memory / OOM | Large models (ViT-L) with high budget | ⬜ Waiting on A |

#### B2. Accelerator Segmenters

| # | Feature | Details | Priority | Status |
|---|---|---|---|---|
| B2.1 | `GradientGuidedSegmenter` | Extract gradient map → skimage watershed → non-uniform static layout | Medium | 🔄 In progress |
| B2.2 | `AdaptiveSegmenter` | Coarse grid → score-driven subdivision → feedback loop. Requires `is_stateful=True` protocol between Imputer ↔ Segmenter | Medium | ⬜ Not started |

#### B3. Masker Extension

| # | Feature | Details | Status |
|---|---|---|---|
| B3.1 | `AttentionMasker` implementation | Hook self-attention, inject -inf mask matrices. Requires PyTorch `register_forward_hook` or HF `output_attentions` override | ⬜ Not started |

#### B4. Backend Adapter Extraction

| # | Feature | Details | Status |
|---|---|---|---|
| B4.1 | `TorchOps` extraction | Move inline PyTorch ops from Imputer/Segmenter into adapter | ⬜ Not started |
| B4.2 | `JaxOps` skeleton | Interface + stub for JAX-native models | ⬜ Not started |

#### B5. Performance Optimization

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
A1.1–A1.8 (migrate experiments)    B2.1 GradientGuidedSegmenter
    │                                   │
    ├─ A2 (equivalence tests) ──────────┤ (bug reports)
    │       │                           │
    │       ▼                           ▼
    ├─ A3 (cross-model) ──────────► B1 (bug fixes)
    │       │                           │
    │       ▼                           ▼
    └─ A4 (feedback) ─────────────► B2.2 AdaptiveSegmenter
                                        │
                                        ▼
                                    B3–B5 (extensions)
```

| Dependency | Blocker | Blocked by |
|---|---|---|
| B1 (bug fix) | A2/A3 reports | Team A findings |
| B2.2 (Adaptive) | `is_stateful` protocol | B1 stability |
| A3 (cross-model) | A1 completion | All experiments pass |

---

### Milestone Schedule

| Week | Team A | Team B | PM Gate |
|---|---|---|---|
| W1 | A1.1–A1.4, A2.1 | B1 (bug fixes), B4.1 (TorchOps) | Experiments 1–4 pass |
| W2 | A1.5–A1.8, A2.2–A2.3 | B2.1 (GradientGuided) | All 8 experiments pass |
| W3 | A3.1–A3.5 (cross-model) | B2.2 (Adaptive), B3.1 | Cross-model tests green |
| W4 | A4 (feedback loop) | B4.2 (JaxOps), B5.1 (memory) | Feature freeze, integration test |

---

### Known Issues (tracked for B1)

- **Crossmodal edge-case processor calls**: When `budget_image % batch_size ≠ 0` or `budget_text % batch_size ≠ 0`, the last image and/or text batch have incomplete sizes (e.g., `img_bs=15, txt_bs=51` for `batch_size=64, budget_image=4559, budget_text=115`). The 2 (img batches) × 2 (text batches) = 4 combinations yield 3 cases where `img_bs ≠ txt_bs`. In those cases `_preprocess_batch()` must re-invoke the HF processor to create inputs with matching batch dimensions. The original `src` code has the same behavior (it calls `processor_function` directly in the equivalent branches), so this is not a regression — it is inherent to the double-loop crossmodal design. Total extra calls per `forward_crossmodal`: at most 3 (~2 ms each, negligible).
- `_repeat_inputs` uses `.expand().clone()` which duplicates memory; could be optimized with stride tricks| ✅ Done (Needs profiling validation)
- No mixed-precision (AMP) support yet — relevant for larger models| ✅ Done (Opt-in, needs numeric validation)
