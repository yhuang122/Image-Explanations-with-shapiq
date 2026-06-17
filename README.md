# Pluggable Image Imputer for shapiq — Demo

Modular vision-language model explanation pipeline built on **shapiq** + **fixlip**.
Pluggable Segmenters (patch, SLIC, gradient-guided) × Maskers (mean, blur, attention)
→ Shapley-interaction explanations for CLIP / SigLIP.

## Environment Setup

### conda

```bash
conda env create -f environment.yml
conda activate shapiq_demo
```

This creates a self-contained environment with all dependencies (PyTorch 2.4.1 + CUDA 11.8,
Transformers 4.51.3, modular shapiq, scikit-image, etc.).

> **Note on reproducibility**: The conda-forge binary of PyTorch differs from the PyPI wheel
> (different cuDNN versions, C++ ABI, and CUDA library bundles).  If you observe tiny
> floating-point differences between runs in different environments, this is expected —
> see the Reproducibility Note below.

---

## Quick Start

```python
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

### Reproducibility Note

`example_siglip_updated.ipynb` interaction values are sensitive to the
PyTorch build.  `conda-forge` and `pip` distribute different binaries of
the same version (`2.4.1`) — they differ in compilation flags, BLAS/LAPACK
backends, and cuDNN bindings.  These produce tiny floating-point
differences in model forward passes (< 1e-6 per coalition), which
**XGBoost proxyshap** amplifies through tree-split decisions.

→ Always use the conda environment above to reproduce the reported numbers.

---

## Notebooks

### `example.ipynb`
CLIP ViT-B/32 + PatchSegmenter + CrossModalMeanMasker — full pipeline.

![example_fixlip](data/report_pictures/example_fixlip.png)

### `example_slic.ipynb`
CLIP ResNet + SLICSegmenter — perceptual superpixels for CNN backbones.

### `example_blur.ipynb`
VisionBlurMasker — Gaussian blur occlusion (CPU skimage).

### `example_gradient_guided.ipynb`
GradientGuidedSegmenter — saliency-guided non-uniform layout.

### `example_siglip_updated.ipynb`
SigLIP model support — see Reproducibility Note above.

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
