# Pluggable Image Imputer for shapiq — Demo

Modular vision-language model explanation pipeline built on **shapiq** + **fixlip**.
Pluggable Segmenters (patch, SLIC, gradient-guided) × Maskers (mean, blur, attention)
→ Shapley-interaction explanations for CLIP / SigLIP.

## Environment Setup

### Option A: uv (recommended)

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
uv sync
source .venv/bin/activate
uv pip install "git+https://github.com/yhuang122/Image-Explanations-with-shapiq.git@shapiq_dev"
```

### Option B: conda (for experiment reproduction)

```bash
conda env create -f environment.yml
conda activate shapiq_demo
```

Legacy conda env (upstream shapiq, for reference): `../fixlip/env_faster.yml`.

---

## Quick Start

```python
import sys
sys.path.insert(0, ".")

from shapiq.imputer.vision import VisionImputerFactory, VisionLanguageGame

factory = VisionImputerFactory()
imputer = factory.build(model, processor, image, text)          # defaults: Patch + CrossModalMean
# imputer = factory.build(model, processor, image, text,
#                         segmenter_config=SegmenterConfig(strategy="slic"),
#                         masker_config=MaskerConfig(strategy="vision_blur"))
game = VisionLanguageGame(imputer, batch_size=64)
# game is now usable with any shapiq approximator (KernelSHAP, SVARM-I, etc.)
```

See notebooks below for end-to-end examples.

---

## Experiment Results

### 1. Faithfulness

How well do Shapley-interaction explanations predict the model's output under coalition
occlusion?

| Script | `experiments/migrated/faithfulness.py` |
|---|---|
| Models | CLIP ViT-B/32, ViT-B/16, ViT-L/14; SigLIP |

| Model | Segmenter | Masker | Pearson r | Spearman ρ | MSE |
|---|---:|---:|---:|---:|---:|
| CLIP ViT-B/32 | patch | crossmodal_mean | — | — | — |

![faithfulness](data/report_pictures/faithfulness.png)

---

### 2. Insertion / Deletion (AID)

Area under the insertion/deletion curve — higher is better.

| Script | `experiments/migrated/insertion_deletion.py` (CLIP) / `*_siglip.py` |
|---|---|

| Model | Segmenter | Masker | AID ↑ | Insertion ↑ | Deletion ↓ |
|---|---:|---:|---:|---:|---:|
| CLIP ViT-B/32 | patch | crossmodal_mean | — | — | — |

![aid_clip](data/report_pictures/insertion_deletion_clip.png)

#### SigLIP

| Model | Segmenter | Masker | AID ↑ |
|---|---:|---:|--|
| SigLIP base-patch16 | patch | crossmodal_mean | — |

![aid_siglip](data/report_pictures/insertion_deletion_siglip.png)

---

### 3. Pointing Game

Positive Gradient Removal (PGR) accuracy — does the explanation correctly identify the
image region most responsible for the prediction?

| Scripts | `experiments/migrated/pointing_game_banzhaf.py`, `*_shapley.py`, `*_crossmodal.py` |
|---|---|

#### Banzhaf Interactions

| Model | Segmenter | PGR ↑ |
|---|---:|---:|
| CLIP ViT-B/32 | patch | — |

#### Shapley Interactions

| Model | Segmenter | PGR ↑ |
|---|---:|---:|
| CLIP ViT-B/32 | patch | — |

#### Crossmodal (Banzhaf)

| Model | Segmenter | PGR ↑ |
|---|---:|---:|
| CLIP ViT-B/32 | patch | — |

![pointing_game](data/report_pictures/pointing_game.png)

---

### 4. Qualitative Examples (MSCOCO)

Example Shapley-interaction explanations on real images.

| Script | `experiments/migrated/explain_mscoco.py` (CLIP) / `*_siglip.py` |
|---|---|

![mscoco_clip](data/report_pictures/mscoco_clip.png)

#### SigLIP

![mscoco_siglip](data/report_pictures/mscoco_siglip.png)

---

### 5. Equivalence — New API vs Legacy

Numerical equivalence between `shapiq.imputer.vision` and the archived `ImputerFactory`.

| Script | `experiments/migrated/test_game_equivalence.py`, `compare_games.py` |
|---|---|

| Metric | Tolerance | Pass? |
|---|---|---|
| `forward_1d` output | Δ < 1e-4 | — |
| `forward_crossmodal` output | Δ < 1e-4 | — |

---

## Notebooks

### `example.ipynb`
CLIP ViT-B/32 + PatchSegmenter + CrossModalMeanMasker — full pipeline.
**Verified**: migrated pipeline produces identical results to the original fixlip implementation.

![example_fixlip](data/report_pictures/example_fixlip.png)

### `example_slic.ipynb`
CLIP ResNet + SLICSegmenter — perceptual superpixels for CNN backbones.

### `example_blur.ipynb`
VisionBlurMasker — Gaussian blur occlusion (CPU skimage).

### `example_gradient_guided.ipynb`
GradientGuidedSegmenter — saliency-guided non-uniform layout.

### `example_siglip.ipynb`
SigLIP model support.

### `example_faster.ipynb`
Batch-size tuning and performance profiling.

### `attention_masker_demo.ipynb`
AttentionMasker — self-attention -inf injection.

### `insertion_deletion.ipynb`
AID curve computation and plotting.

---

## Adding Results

1. Run the experiment script, save the output figure to `data/report_pictures/`.
2. Update the relevant section above with numeric metrics or the image.
3. Add the image using:
   ```markdown
   ![label](data/report_pictures/filename.png)
   ```
